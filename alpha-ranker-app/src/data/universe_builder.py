"""
Universe construction for large multi-asset research universes.

The `UniverseBuilder` supports:
- Named universes (e.g. 'msci_world') – pluggable in the future
- Custom ISIN lists from CSV
- Legacy ticker-based universes via `core.data.fetch_universe`
- Auto-fill: fetch large-cap candidates and add until target count (with ISIN resolution)

ISINs are treated as the primary identifier; tickers are resolved via the
ISIN mapper where possible, with graceful degradation when mappings are
unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import isin_mapper


@dataclass
class Universe:
    """Container for an investable universe."""

    isins: List[str]
    tickers: List[str]


class UniverseBuilder:
    """High-level utility for building large universes of 3k–10k+ names."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent.parent.parent / "data" / "universes"

    # Public API -----------------------------------------------------
    def build(
        self,
        source: str,
        identifiers: str = "isin",
        isin_file: Optional[str] = None,
    ) -> Universe:
        """
        Build a universe.

        Examples:
        - UniverseBuilder().build(source="msci_world", identifiers="isin")
        - UniverseBuilder().build(source="custom_list", isin_file="global_equities.csv")
        """
        source = source.lower()
        if source == "custom_list":
            if not isin_file:
                raise ValueError("isin_file is required for source='custom_list'")
            return self._from_custom_isin_file(isin_file)

        if source == "msci_world":
            return self._from_named_file("msci_world_isins.csv")

        # Fallback: use legacy ticker-based universe (S&P + extras)
        try:
            from core.data import fetch_universe

            tickers, _, _ = fetch_universe()
            tickers = list(tickers)
            # Optional: resolve tickers to ISINs and warn on low coverage
            try:
                from .isin_mapper import batch_map_tickers_to_isin, report_isin_coverage
                ticker_to_isin, source_counts = batch_map_tickers_to_isin(tickers)
                isins = [ticker_to_isin.get(t) or "" for t in tickers]
                isins = [i for i in isins if i]
                resolved = len(isins)
                total = len(tickers)
                if report_isin_coverage and total > 0:
                    report_isin_coverage(total, resolved, source_counts)
                if total >= 50 and resolved < 0.8 * total:
                    import logging
                    logging.getLogger(__name__).warning(
                        "ISIN coverage %.0f%% < 80%%. Consider adding OPENFIGI_API_KEY or static_isins.csv.",
                        (resolved / total * 100),
                    )
            except Exception:
                pass
            return Universe(isins=[], tickers=tickers)
        except Exception:
            return Universe(isins=[], tickers=[])

    # Internal helpers -----------------------------------------------
    def _from_custom_isin_file(self, filename: str) -> Universe:
        path = Path(filename)
        if not path.is_absolute():
            path = self.base_path / path
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        col = None
        for c in df.columns:
            if c.lower() in ("isin", "isins"):
                col = c
                break
        if col is None:
            raise ValueError(f"No ISIN column found in {path}")

        isins = [str(v).strip().upper() for v in df[col].dropna().unique().tolist()]
        tickers = []
        for isin in isins:
            ident = isin_mapper.resolve_asset_identifier(isin)
            if ident:
                tickers.append(ident.ticker)
        return Universe(isins=isins, tickers=tickers)

    def _from_named_file(self, filename: str) -> Universe:
        path = self.base_path / filename
        if not path.exists():
            # Graceful degradation: empty ISIN list, rely on tickers elsewhere
            return Universe(isins=[], tickers=[])
        return self._from_custom_isin_file(str(path))

