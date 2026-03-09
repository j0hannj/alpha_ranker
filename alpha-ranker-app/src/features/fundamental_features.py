"""
Fundamental + technical feature construction (point-in-time).

This is a direct extraction of the logic from `core.model`, with only
minimal reshaping so it can be reused by multiple training pipelines
and backtest engines.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def get_fundamentals_asof(ticker_data: List[dict], as_of_date) -> Optional[dict]:
    """Get MOST RECENT fundamentals filed BEFORE as_of_date. Enforces temporal consistency."""
    as_of = pd.Timestamp(as_of_date)
    valid = [r for r in ticker_data if pd.Timestamp(r["filing_date"]) <= as_of]
    if not valid:
        return None
    latest = valid[-1]
    last_4 = valid[-4:] if len(valid) >= 4 else valid
    ttm = {}
    for f in ["revenue", "net_income", "ebitda"]:
        vals = [r[f] for r in last_4 if r.get(f) is not None]
        if vals:
            ttm[f"{f}_ttm"] = sum(vals)
    if len(valid) >= 8:
        prev_4 = valid[-8:-4]
        rn = sum(r["revenue"] for r in last_4 if r.get("revenue"))
        rp = sum(r["revenue"] for r in prev_4 if r.get("revenue"))
        if rp > 0:
            ttm["revenue_growth_yoy"] = rn / rp - 1
        en = sum(r["eps"] for r in last_4 if r.get("eps"))
        ep = sum(r["eps"] for r in prev_4 if r.get("eps"))
        if ep != 0:
            ttm["eps_growth_yoy"] = en / ep - 1
    return {**latest, **ttm}


def build_features_asof(prices, fundamentals_db: Dict[str, List[dict]], macro: Dict, as_of_date, tickers: Iterable[str]) -> pd.DataFrame:
    """Build feature matrix using ONLY data available at as_of_date."""
    as_of = pd.Timestamp(as_of_date)
    records: List[dict] = []
    for ticker in tickers:
        row: Dict = {"ticker": ticker, "date": str(as_of_date)}
        # Fundamental features (point-in-time via filing date)
        td = fundamentals_db.get(ticker)
        if td:
            fund = get_fundamentals_asof(td, as_of_date)
            if fund:
                for f in [
                    "pe_ratio",
                    "pb_ratio",
                    "ev_ebitda",
                    "roe",
                    "debt_to_equity",
                    "current_ratio",
                    "gross_margin",
                    "operating_margin",
                    "net_margin",
                    "peg_ratio",
                    "dividend_yield",
                    "fcf_per_share",
                ]:
                    row[f] = fund.get(f)
                mcap = fund.get("market_cap")
                if mcap and mcap > 0:
                    row["log_market_cap"] = np.log(mcap)
                for f in ["revenue_ttm", "net_income_ttm", "revenue_growth_yoy", "eps_growth_yoy"]:
                    row[f] = fund.get(f)
                if fund.get("fcf_per_share") and mcap and mcap > 0:
                    row["fcf_yield"] = fund["fcf_per_share"] * 1e6 / mcap
        # Price/technical features (using data up to as_of only)
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices[(ticker, "Close")].dropna()
                close = close[close.index <= as_of]
            else:
                close = pd.Series()
            if len(close) >= 60:
                cur = close.iloc[-1]
                if len(close) > 21:
                    row["return_1m"] = cur / close.iloc[-21] - 1
                if len(close) > 63:
                    row["return_3m"] = cur / close.iloc[-63] - 1
                if len(close) > 126:
                    row["return_6m"] = cur / close.iloc[-126] - 1
                if len(close) > 252:
                    row["return_12m"] = cur / close.iloc[-252] - 1
                    row["momentum_12_1"] = close.iloc[-21] / close.iloc[-252] - 1
                daily = close.pct_change().dropna()
                row["volatility_1m"] = daily.tail(21).std() * np.sqrt(252)
                row["volatility_3m"] = daily.tail(63).std() * np.sqrt(252)
                if len(daily) > 252:
                    row["volatility_12m"] = daily.tail(252).std() * np.sqrt(252)
                    if row["volatility_12m"] > 0:
                        row["sharpe_12m"] = row.get("return_12m", 0) / row["volatility_12m"]
                ma50 = close.tail(50).mean()
                if ma50 > 0:
                    row["price_vs_ma50"] = cur / ma50 - 1
                if len(close) > 200:
                    ma200 = close.tail(200).mean()
                    if ma200 > 0:
                        row["price_vs_ma200"] = cur / ma200 - 1
                row["drawdown_from_high"] = cur / close.tail(min(252, len(close))).max() - 1
                row["distance_from_low"] = cur / close.tail(min(252, len(close))).min() - 1
        except Exception:
            pass
        # Macro features (contemporaneous)
        if isinstance(macro, dict):
            for k, v in macro.items():
                if isinstance(v, (int, float)):
                    row[f"macro_{k}"] = v
        records.append(row)
    return pd.DataFrame(records)

