"""
Alpha Vantage client.

Used as an additional, optional source for prices or FX when the primary
providers are unavailable. This keeps the interface minimal; only functions
actually needed by the research platform are implemented.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Iterable, Optional

import pandas as pd


class AlphaVantageClient:
    """Optional Alpha Vantage price client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_daily_prices(self, symbol: str) -> Optional[pd.Series]:
        """
        Fetch daily adjusted close prices for a single symbol.

        NOTE: Alpha Vantage has low rate limits; this is intended as a
        fallback source, not the primary engine for large universes.
        """
        if not self.available:
            return None

        try:
            import requests  # type: ignore
        except Exception:
            return None

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json().get("Time Series (Daily)", {})
            if not data:
                return None
            records = [
                (datetime.strptime(d, "%Y-%m-%d"), float(v["5. adjusted close"]))
                for d, v in data.items()
            ]
            records.sort(key=lambda x: x[0])
            idx = [d for d, _ in records]
            vals = [p for _, p in records]
            return pd.Series(vals, index=idx, name=symbol)
        except Exception:
            return None

