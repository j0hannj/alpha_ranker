"""
Tiingo client (optional).

Provides an additional data source for historical prices. Intended to be used
by the `DataPipeline` as a fallback when primary sources are unavailable.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd


class TiingoClient:
    """Optional Tiingo price client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_daily_prices(self, symbol: str) -> Optional[pd.Series]:
        """
        Fetch daily adjusted close prices for a single symbol.
        """
        if not self.available:
            return None

        try:
            import requests  # type: ignore
        except Exception:
            return None

        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {
            "startDate": "2015-01-01",
            "resampleFreq": "daily",
            "token": self.api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            idx = [datetime.fromisoformat(row["date"][:10]) for row in data]
            vals = [float(row.get("adjClose", row.get("close"))) for row in data]
            return pd.Series(vals, index=idx, name=symbol)
        except Exception:
            return None

