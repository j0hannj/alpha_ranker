"""
Polygon.io client (optional).

Provides an interface for US equities and indices where an API key is available.
Designed to be a secondary/tertiary source, not a hard dependency.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Iterable, Optional

import pandas as pd


class PolygonClient:
    """Optional Polygon.io price client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_daily_prices(self, symbol: str) -> Optional[pd.Series]:
        """
        Fetch daily close prices for a single symbol using Polygon aggregates.
        """
        if not self.available:
            return None

        try:
            import requests  # type: ignore
        except Exception:
            return None

        # Simple example: last ~5 years of daily bars
        end = datetime.now().strftime("%Y-%m-%d")
        # Polygon requires a start; use a far back date to keep implementation simple
        start = "2015-01-01"
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json().get("results", [])
            if not data:
                return None
            idx = [datetime.fromtimestamp(row["t"] / 1000.0) for row in data]
            vals = [float(row["c"]) for row in data]
            return pd.Series(vals, index=idx, name=symbol)
        except Exception:
            return None

