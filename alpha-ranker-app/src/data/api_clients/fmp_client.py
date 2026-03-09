"""
Financial Modeling Prep (FMP) API client.

This wraps the existing FMP usage in `core.data` and `core.model` and exposes
a simple class-based interface suitable for composing in the `DataPipeline`.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

from .. import isin_mapper  # type: ignore  # local package import

try:
    # Reuse existing FMP helpers for robustness
    from core.model import fetch_all_fundamentals as _fetch_all_fundamentals_model
except Exception:
    _fetch_all_fundamentals_model = None  # type: ignore

try:
    from core import data as core_data
except Exception:  # pragma: no cover - defensive
    core_data = None  # type: ignore


class FMPClient:
    """Client for FMP fundamentals, ratios and (optionally) prices."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_fundamentals_bulk(
        self,
        tickers: Iterable[str],
        callback=None,
    ) -> Dict[str, List[dict]]:
        """
        Bulk fundamentals + ratios per ticker.

        Returns the same structure that `core.model.fetch_all_fundamentals`
        currently returns when available, otherwise falls back to the more
        granular helpers in `core.data`.
        """
        if not self.available:
            return {}

        if _fetch_all_fundamentals_model is not None:
            return _fetch_all_fundamentals_model(list(tickers), self.api_key, callback)

        # Fallback: call through core.data helpers per ticker
        result: Dict[str, List[dict]] = {}
        if core_data is None:
            return result

        for t in tickers:
            try:
                inc = core_data.fetch_fundamentals(t, self.api_key)
                ratios = core_data.fetch_ratios(t, self.api_key)
                merged: List[dict] = []
                # Merge income + ratios by date where possible
                by_date: Dict[str, dict] = {}
                for row in inc:
                    d = row.get("date")
                    if not d:
                        continue
                    by_date.setdefault(d, {}).update(row)
                for row in ratios:
                    d = row.get("date")
                    if not d:
                        continue
                    by_date.setdefault(d, {}).update(row)
                for d, row in sorted(by_date.items()):
                    row["ticker"] = t
                    merged.append(row)
                if merged:
                    result[t] = merged
            except Exception:
                continue
        return result

    def fetch_macro(self):
        """Delegate to the macro fetcher in `core.data` where possible."""
        if core_data is not None:
            return core_data.fetch_macro()
        return {}

