"""
Yahoo Finance client.

This mirrors and extends the logic in `core.data.fetch_universe` and
`core.data.fetch_price` to provide a reusable client for:
- historical prices
- basic fundamentals / metadata
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


class YahooClient:
    """Thin wrapper around `yfinance` for prices and basic fundamentals."""

    def __init__(self, years: int = 5):
        self.years = years

    @staticmethod
    def _load_yfinance():
        import yfinance as yf  # type: ignore

        return yf

    def fetch_prices_and_fundamentals(
        self, tickers: Iterable[str]
    ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """Fetch multi-ticker price history and basic fundamentals."""
        yf = self._load_yfinance()
        tickers = list(set(tickers))
        end = datetime.now()
        start = end - timedelta(days=self.years * 365)

        prices = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )

        fundamentals: Dict[str, dict] = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                if not info:
                    continue
                fundamentals[t] = info
            except Exception:
                continue
        return prices, fundamentals

    def fetch_realtime_prices(self, tickers: Iterable[str]) -> Dict[str, dict]:
        """Fetch latest prices for a set of tickers."""
        yf = self._load_yfinance()
        out: Dict[str, dict] = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                p = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose")
                )
                if p is None:
                    continue
                out[t] = {
                    "ticker": t,
                    "price": float(p),
                    "currency": info.get("currency", "USD"),
                    "name": info.get("shortName", t),
                    "sector": info.get("sector"),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception:
                continue
        return out

