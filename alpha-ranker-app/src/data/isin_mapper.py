"""
ISIN mapping utilities for the research platform.

This module sits on top of `core.isin` and exposes a clean, reusable API:

- map_isin_to_ticker(isin) -> ticker or None
- map_ticker_to_isin(ticker) -> isin or None (multi-source chain)
- resolve_asset_identifier(raw_id) -> dict with ticker/isin/exchange

Ticker -> ISIN uses a resolution chain: local_cache -> static_file -> OpenFIGI -> yfinance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from core.isin import resolve_isin as _resolve_isin_core
except Exception:  # pragma: no cover - defensive
    _resolve_isin_core = None  # type: ignore

try:
    from .isin_resolution_chain import (
        normalize_ticker,
        resolve_ticker_to_isin as _resolve_ticker_to_isin,
        resolve_ticker_to_isin_with_source,
        batch_resolve_tickers_to_isin,
        report_isin_coverage,
        RESOLUTION_CHAIN,
    )
except Exception:
    _resolve_ticker_to_isin = None  # type: ignore
    resolve_ticker_to_isin_with_source = None  # type: ignore
    batch_resolve_tickers_to_isin = None  # type: ignore
    report_isin_coverage = None  # type: ignore
    normalize_ticker = lambda t: (t or "", None)  # type: ignore
    RESOLUTION_CHAIN = []  # type: ignore


@dataclass
class AssetIdentifier:
    """Canonical representation of a single asset identifier."""

    ticker: str
    isin: Optional[str] = None
    exchange: Optional[str] = None
    source: Optional[str] = None


def map_isin_to_ticker(isin: str) -> Optional[str]:
    """Best-effort mapping from ISIN to ticker, using `core.isin`."""
    if not _resolve_isin_core:
        return None
    res = _resolve_isin_core(isin)
    if not res:
        return None
    return res.get("ticker")


def map_ticker_to_isin(ticker: str) -> Optional[str]:
    """
    Resolve ticker to ISIN via multi-source chain:
    local_cache (SQLite) -> static_isins.csv -> OpenFIGI -> yfinance.
    Results are persisted to the shared cache.
    """
    if _resolve_ticker_to_isin:
        return _resolve_ticker_to_isin(ticker, use_chain=True)

    # Fallback: legacy cache scan (core.isin cache keyed by ISIN)
    from pathlib import Path
    import json
    try:
        from core.isin import CACHE_FILE
        if Path(CACHE_FILE).exists():
            data = json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
            for isin, info in data.items():
                if info.get("ticker") == ticker:
                    return isin
    except Exception:
        pass
    return None


def resolve_asset_identifier(raw: str) -> Optional[AssetIdentifier]:
    """
    Resolve a user-provided identifier (ticker or ISIN) into a canonical form.

    - If it looks like an ISIN (12 chars, alpha prefix), try ISIN→ticker.
    - Otherwise treat as ticker and resolve ISIN via multi-source chain.
    """
    raw = raw.strip().upper()
    if not raw:
        return None

    # ISIN pattern: 2 letters + 10 alphanumerics
    if len(raw) == 12 and raw[:2].isalpha():
        ticker = map_isin_to_ticker(raw) or raw
        return AssetIdentifier(ticker=ticker, isin=raw, source="isin")

    # Otherwise assume ticker, resolve ISIN via chain
    isin = map_ticker_to_isin(raw)
    return AssetIdentifier(ticker=raw, isin=isin, source="ticker")


def batch_map_tickers_to_isin(
    tickers: List[str],
    callback=None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Resolve multiple tickers to ISINs; returns (ticker_to_isin, source_counts).
    Use report_isin_coverage(total, resolved, source_counts) after to log.
    """
    if batch_resolve_tickers_to_isin:
        return batch_resolve_tickers_to_isin(tickers, callback=callback)
    result = {}
    for t in tickers:
        isin = map_ticker_to_isin(t)
        if isin:
            result[t] = isin
    return result, {}

