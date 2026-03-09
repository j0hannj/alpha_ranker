"""
Portfolio builder: orchestrates confidence filter and allocation engine.

Pipeline:
  1. Filter assets by confidence threshold (LOW / MEDIUM / HIGH).
  2. Rank by alpha_score.
  3. Allocate capital proportionally (equal / risk parity / alpha weight).
  4. Convert allocation to integer number of shares using real asset price.
  5. Ensure total cost never exceeds budget.

Output format: list of dicts with ticker, price, units, invested_amount,
confidence, alpha_score (and optional name, sector, reason for UI).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .confidence_filter import ConfidenceFilter, ConfidenceLevel, CONFIDENCE_THRESHOLDS
from .allocation_engine import AllocationEngine


def build_suggested_portfolio(
    model_results: pd.DataFrame,
    budget: float,
    confidence_level: ConfidenceLevel = "MEDIUM",
    weight_method: str = "alpha_weight",
    max_positions: Optional[int] = None,
    etf_budget: float = 0.0,
    etf_positions: Optional[List[tuple]] = None,
) -> List[dict]:
    """
    Build a suggested portfolio from alpha model results.

    - model_results: DataFrame with columns at least: ticker, confidence,
      alpha_score, current_price; optional: name, sector, volatility_12m.
    - budget: total capital (e.g. EUR).
    - confidence_level: LOW | MEDIUM | HIGH (filters by confidence).
    - weight_method: equal_weight | risk_parity | alpha_weight.
    - max_positions: cap on number of stock picks (ETFs are additional).
    - etf_budget: amount reserved for ETFs (subtracted from budget for stocks).
    - etf_positions: optional list of (ticker, name, weight) for ETF slice.

    Returns list of position dicts with ticker, price, units, invested_amount,
    confidence, alpha_score, and optional name, sector, reason, src.
    """
    stock_budget = budget - etf_budget
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
                    out.append({
                        "src": "ETF",
                        "ticker": ticker,
                        "name": name,
                        "sector": "Global",
                        "price": round(price, 2),
                        "units": units,
                        "invested_amount": round(invested, 2),
                        "alloc": round(invested, 2),
                        "confidence": None,
                        "alpha_score": None,
                        "reason": "Core diversification",
                    })
            else:
                out.append({
                    "src": "ETF",
                    "ticker": ticker,
                    "name": name,
                    "sector": "Global",
                    "price": None,
                    "units": 0,
                    "invested_amount": 0,
                    "alloc": round(alloc, 2),
                    "confidence": None,
                    "alpha_score": None,
                    "reason": "Core diversification (price unknown)",
                })

    # Stock slice: filter by confidence, then allocate
    if stock_budget > 0 and model_results is not None and not model_results.empty:
        df = ConfidenceFilter(confidence_level).filter(model_results)
        engine = AllocationEngine(
            budget=stock_budget,
            weight_method=weight_method,
            max_positions=max_positions,
            allow_fractional_shares=False,
        )
        # Map alpha_score for allocation (use predicted_return_pct if alpha_score missing)
        if "alpha_score" not in df.columns and "predicted_return_pct" in df.columns:
            df = df.copy()
            df["alpha_score"] = df["predicted_return_pct"] / 100.0
        positions = engine.allocate(
            df,
            price_col="current_price",
            alpha_col="alpha_score",
            confidence_col="confidence",
        )
        for p in positions:
            row = model_results[model_results["ticker"] == p["ticker"]]
            name = row.iloc[0]["name"] if not row.empty and "name" in row.columns else p["ticker"]
            sector = row.iloc[0]["sector"] if not row.empty and "sector" in row.columns else ""
            p["src"] = "Model"
            p["name"] = name[:22] if isinstance(name, str) else str(name)[:22]
            p["sector"] = sector[:14] if isinstance(sector, str) else ""
            p["alloc"] = p["invested_amount"]
            p["reason"] = f"Rank signal, confidence {p.get('confidence'):.2f}" if p.get("confidence") is not None else "Rank signal"
            out.append(p)

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
