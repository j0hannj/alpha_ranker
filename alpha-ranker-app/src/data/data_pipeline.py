"""
Unified data ingestion pipeline.

This module orchestrates multiple API clients and universe builders to provide
the full input bundle required by the alpha research engine:

    prices, fundamentals, macro, fx, sentiment, metadata

It is designed to:
- Support many providers simultaneously (Yahoo, FMP, Alpha Vantage, Polygon, Tiingo, etc.)
- Try multiple sources when data is missing
- Scale to universes of several thousand securities

For backward compatibility, `core.data.fetch_all_data` remains the primary
entrypoint used by the desktop app; new research code can consume this
pipeline directly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .api_clients import (
    AlphaVantageClient,
    FMPClient,
    PolygonClient,
    TiingoClient,
    YahooClient,
)
from .universe_builder import UniverseBuilder


@dataclass
class DataBundle:
    """Canonical data bundle consumed by the model layer."""

    tickers: List[str]
    prices: pd.DataFrame
    fundamentals: Dict[str, dict]
    macro: Dict[str, float]
    fx: Dict[str, float]
    sentiment: Dict[str, float]
    fetched_at: str


class DataPipeline:
    """
    High-level orchestrator for multi-source data ingestion.

    Typical usage:

        pipeline = DataPipeline()
        bundle = pipeline.load_full_universe("msci_world")
    """

    def __init__(self):
        self.universe_builder = UniverseBuilder()
        self.yahoo = YahooClient()
        self.fmp = FMPClient()
        self.alpha_vantage = AlphaVantageClient()
        self.polygon = PolygonClient()
        self.tiingo = TiingoClient()

    # Public API -----------------------------------------------------
    def load_full_universe(
        self,
        source: str = "msci_world",
        identifiers: str = "isin",
        extra_tickers: Optional[Iterable[str]] = None,
    ) -> DataBundle:
        """
        Build a universe, fetch all data, and return a `DataBundle`.
        """
        uni = self.universe_builder.build(source=source, identifiers=identifiers)
        tickers = list(set(uni.tickers + list(extra_tickers or [])))

        # 1) Prices + basic fundamentals from Yahoo (primary)
        prices, yf_fund = self.yahoo.fetch_prices_and_fundamentals(tickers)

        # 2) Fundamentals from FMP (if available)
        if self.fmp.available:
            fmp_fundamentals = self.fmp.fetch_fundamentals_bulk(tickers)
        else:
            fmp_fundamentals = {}

        fundamentals = self._merge_fundamentals(yf_fund, fmp_fundamentals)

        # 3) Macro from FRED via existing helper
        macro = self.fmp.fetch_macro() if self.fmp.available else {}

        # 4) FX – delegate to existing core.data implementation where possible
        fx = self._fetch_fx_with_fallback()

        # 5) Sentiment – still delegated to the existing news layer for now
        sentiment = self._fetch_sentiment_with_fallback(list(fundamentals.keys()))

        return DataBundle(
            tickers=tickers,
            prices=prices,
            fundamentals=fundamentals,
            macro=macro,
            fx=fx,
            sentiment=sentiment,
            fetched_at=datetime.now().isoformat(),
        )

    # Internal helpers -----------------------------------------------
    def _merge_fundamentals(
        self,
        yahoo_fund: Dict[str, dict],
        fmp_fund: Dict[str, List[dict]],
    ) -> Dict[str, dict]:
        """
        Merge Yahoo static fundamentals with time-series FMP data.

        For now we simply expose Yahoo's latest snapshot and attach FMP
        time series under a `"fmp_quarterly"` key to avoid breaking the
        existing modeling code which already knows how to consume the
        FMP format.
        """
        out: Dict[str, dict] = {}
        for t, info in yahoo_fund.items():
            base = dict(info)
            if t in fmp_fund:
                base["fmp_quarterly"] = fmp_fund[t]
            out[t] = base
        for t, rows in fmp_fund.items():
            if t not in out:
                out[t] = {"fmp_quarterly": rows}
        return out

    def _fetch_fx_with_fallback(self) -> Dict[str, float]:
        """Use the existing FX helper where possible."""
        try:
            from core.data import fetch_fx

            return fetch_fx()
        except Exception:
            return {}

    def _fetch_sentiment_with_fallback(self, tickers: List[str]) -> Dict[str, float]:
        """
        Delegates to the existing `core.news.batch_sentiment` where available.

        News sentiment APIs (e.g. RavenPack, Aylien, etc.) can later be wired
        in here with dedicated clients.
        """
        try:
            from core.news import batch_sentiment

            return batch_sentiment(tickers[:150], callback=None)  # reuse existing cap
        except Exception:
            return {}

