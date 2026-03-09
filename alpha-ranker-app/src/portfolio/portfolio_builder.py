"""
Portfolio builder: orchestrates confidence filter and allocation engine.

Pipeline:
  1. Filter assets by confidence threshold (LOW / MEDIUM / HIGH).
  2. Rank by alpha_score.
  3. Allocate capital proportionally (equal / risk parity / alpha weight).
  4. Convert allocation to integer number of shares using real asset price.
  5. Ensure total cost never exceeds budget.

Output includes investment horizon and exit strategy per recommendation:
  ticker, current_price, units_to_buy, investment_amount, confidence, expected_return,
  expected_holding_period, strategy_type, target_price, stop_loss, review_date, model_consensus_score.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from .confidence_filter import ConfidenceFilter, ConfidenceLevel, CONFIDENCE_THRESHOLDS
from .allocation_engine import AllocationEngine
from .transaction_cost_model import (
    TransactionCostParams,
    estimate_transaction_cost,
    is_trade_economically_viable,
    expected_return_as_fraction,
)

try:
    from core.engine_config import get_portfolio_settings, strategy_type_from_horizon
except Exception:
    get_portfolio_settings = None
    strategy_type_from_horizon = lambda d: "MEDIUM_TERM" if d <= 180 else "LONG_TERM"


def build_suggested_portfolio(
    model_results: pd.DataFrame,
    budget: float,
    confidence_level: ConfidenceLevel = "MEDIUM",
    weight_method: str = "alpha_weight",
    max_positions: Optional[int] = None,
    etf_budget: float = 0.0,
    etf_positions: Optional[List[tuple]] = None,
    transaction_cost_params: Optional[TransactionCostParams] = None,
    min_return_vs_cost_multiple: float = 3.0,
    holding_horizon_days: Optional[int] = None,
    default_stop_loss_pct: float = 10.0,
) -> List[dict]:
    """
    Build a suggested portfolio from alpha model results.
    Rejects buys where expected_return < min_return_vs_cost_multiple * transaction_cost.
    Each recommendation includes: ticker, current_price, units_to_buy, investment_amount,
    confidence, expected_return, expected_holding_period, strategy_type, target_price,
    stop_loss, review_date, model_consensus_score.
    """
    stock_budget = budget - etf_budget
    tx_params = transaction_cost_params or TransactionCostParams()
    ps = get_portfolio_settings() if get_portfolio_settings else {}
    horizon = holding_horizon_days or ps.get("holding_horizon_days", 365)
    review_frequency_days = ps.get("review_frequency_days", 30)
    if stock_budget < 0:
        stock_budget = 0

    out: List[dict] = []

    # ETF slice: fixed allocations; use real price for units
    if etf_budget > 0 and etf_positions:
        for ticker, name, weight in etf_positions:
            alloc = etf_budget * weight
            if alloc < 50:
                continue
            # Price must be resolved from model_results or left for UI to fill
            price = None
            if model_results is not None and not model_results.empty:
                m = model_results[model_results["ticker"] == ticker]
                if not m.empty and pd.notna(m.iloc[0].get("current_price")):
                    price = float(m.iloc[0]["current_price"])
            if price and price > 0:
                units = int(alloc / price)
                if units > 0:
                    invested = units * price
                    tx_cost = estimate_transaction_cost(price, units, notional=invested, params=tx_params)
                    review_date = (datetime.now() + timedelta(days=review_frequency_days)).strftime("%Y-%m-%d")
                    out.append({
                        "src": "ETF",
                        "ticker": ticker,
                        "name": name,
                        "sector": "Global",
                        "price": round(price, 2),
                        "current_price": round(price, 2),
                        "units": units,
                        "units_to_buy": units,
                        "invested_amount": round(invested, 2),
                        "investment_amount": round(invested, 2),
                        "alloc": round(invested, 2),
                        "confidence": None,
                        "alpha_score": None,
                        "expected_return": None,
                        "transaction_cost": round(tx_cost, 2),
                        "target_price": None,
                        "stop_loss": None,
                        "holding_horizon": horizon,
                        "expected_holding_period": horizon,
                        "strategy_type": "LONG_TERM",
                        "review_date": review_date,
                        "model_consensus_score": None,
                        "reason": "Core diversification",
                    })
            else:
                out.append({
                    "src": "ETF",
                    "ticker": ticker,
                    "name": name,
                    "sector": "Global",
                    "price": None,
                    "current_price": None,
                    "units": 0,
                    "units_to_buy": 0,
                    "invested_amount": 0,
                    "investment_amount": 0,
                    "alloc": round(alloc, 2),
                    "confidence": None,
                    "alpha_score": None,
                    "expected_return": None,
                    "transaction_cost": None,
                    "target_price": None,
                    "stop_loss": None,
                    "holding_horizon": horizon,
                    "expected_holding_period": horizon,
                    "strategy_type": "LONG_TERM",
                    "review_date": None,
                    "model_consensus_score": None,
                    "reason": "Core diversification (price unknown)",
                })

    # Stock slice: filter by confidence, then allocate. Fallbacks to avoid empty candidate list.
    if stock_budget > 0 and model_results is not None and not model_results.empty:
        df = ConfidenceFilter(confidence_level).filter(model_results)
        # If confidence filter removed everyone, retry with LOW so we still get candidates when signals exist
        if df.empty and len(model_results) > 0:
            df = ConfidenceFilter("LOW").filter(model_results)
        if "alpha_score" not in df.columns and "predicted_return_pct" in df.columns:
            df = df.copy()
            df["alpha_score"] = df["predicted_return_pct"] / 100.0
        engine = AllocationEngine(
            budget=stock_budget,
            weight_method=weight_method,
            max_positions=max_positions,
            allow_fractional_shares=False,
        )
        positions = engine.allocate(
            df,
            price_col="current_price",
            alpha_col="alpha_score",
            confidence_col="confidence",
        )
        # If allocation returned 0 (e.g. no valid prices), try alternate price column if present
        if not positions and not df.empty:
            for col in ("target_mean", "price"):
                if col in df.columns and df[col].notna().any():
                    df = df.copy()
                    df["current_price"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                    df = df[df["current_price"] > 0]
                    if not df.empty:
                        positions = engine.allocate(df, price_col="current_price", alpha_col="alpha_score", confidence_col="confidence")
                    break
        def _add_position(
            p: dict,
            name: str,
            sector: str,
            price: float,
            invested: float,
            units: int,
            alpha_val: Optional[float],
            expected_return_frac: float,
            tx_cost: float,
            horizon: int,
            default_stop_pct: float,
            review_date: str,
            row: pd.DataFrame,
            mult_used: float,
        ) -> None:
            target_price = round(price * (1 + expected_return_frac), 2) if expected_return_frac else None
            stop_loss = round(price * (1 - default_stop_pct), 2) if price else None
            strategy_type = strategy_type_from_horizon(horizon)
            model_consensus_score = None
            if not row.empty:
                for col in ("reliability_score", "model_agreement_score"):
                    if col in row.columns and pd.notna(row.iloc[0].get(col)):
                        model_consensus_score = round(float(row.iloc[0][col]), 3)
                        break
            p["src"] = "Model"
            p["name"] = name[:22] if isinstance(name, str) else str(name)[:22]
            p["sector"] = sector[:14] if isinstance(sector, str) else ""
            p["alloc"] = invested
            p["expected_return"] = round(expected_return_frac * 100, 2) if expected_return_frac else None
            p["transaction_cost"] = round(tx_cost, 2)
            p["target_price"] = target_price
            p["stop_loss"] = stop_loss
            p["holding_horizon"] = horizon
            p["expected_holding_period"] = horizon
            p["strategy_type"] = strategy_type
            p["review_date"] = review_date
            p["model_consensus_score"] = model_consensus_score
            p["current_price"] = price
            p["units_to_buy"] = units
            p["investment_amount"] = invested
            p["reason"] = f"Rank signal, confidence {p.get('confidence'):.2f}" if p.get("confidence") is not None else "Rank signal"
            if mult_used < min_return_vs_cost_multiple:
                p["reason"] = (p.get("reason", "") or "") + " (relaxed tx)"
            out.append(p)

        relaxed_multiple = 2.0  # fallback when 3x would yield zero candidates
        for min_mult in (min_return_vs_cost_multiple, relaxed_multiple):
            added_this_pass = 0
            for p in positions:
                row = model_results[model_results["ticker"] == p["ticker"]]
                name = row.iloc[0]["name"] if not row.empty and "name" in row.columns else p["ticker"]
                sector = row.iloc[0]["sector"] if not row.empty and "sector" in row.columns else ""
                price = p["price"]
                invested = p["invested_amount"]
                units = p["units"]
                alpha_val = p.get("alpha_score")
                if alpha_val is None and not row.empty:
                    pr = row.iloc[0].get("predicted_return_pct")
                    alpha_val = float(pr) / 100.0 if pr is not None and pd.notna(pr) else None
                expected_return_frac = expected_return_as_fraction(alpha_val * 100) if alpha_val is not None else 0.0
                tx_cost = estimate_transaction_cost(price, units, notional=invested, params=tx_params)
                tx_cost_frac = tx_cost / invested if invested and invested > 0 else 0
                if not is_trade_economically_viable(expected_return_frac, tx_cost_frac, min_multiple=min_mult):
                    continue
                # Skip if we already added this ticker in a previous pass
                if any(o.get("ticker") == p["ticker"] and o.get("src") == "Model" for o in out):
                    continue
                review_date = (datetime.now() + timedelta(days=review_frequency_days)).strftime("%Y-%m-%d")
                stop_pct = default_stop_loss_pct / 100.0
                _add_position(
                    p, name, sector, price, invested, units, alpha_val,
                    expected_return_frac, tx_cost, horizon, stop_pct, review_date, row, min_mult,
                )
                added_this_pass += 1
            if added_this_pass > 0:
                break

    return out


class PortfolioBuilder:
    """Convenience wrapper around build_suggested_portfolio with stored defaults."""

    def __init__(
        self,
        confidence_level: ConfidenceLevel = "MEDIUM",
        weight_method: str = "alpha_weight",
        max_positions: Optional[int] = 10,
    ):
        self.confidence_level = confidence_level
        self.weight_method = weight_method
        self.max_positions = max_positions

    def build(
        self,
        model_results: pd.DataFrame,
        budget: float,
        etf_pct: float = 0.3,
        etf_positions: Optional[List[tuple]] = None,
    ) -> List[dict]:
        etf_budget = budget * etf_pct
        if etf_positions is None:
            etf_positions = [
                ("IWDA.AS", "iShares MSCI World", 0.6),
                ("VWCE.DE", "Vanguard All-World", 0.4),
            ]
        return build_suggested_portfolio(
            model_results=model_results,
            budget=budget,
            confidence_level=self.confidence_level,
            weight_method=self.weight_method,
            max_positions=self.max_positions,
            etf_budget=etf_budget,
            etf_positions=etf_positions,
        )
