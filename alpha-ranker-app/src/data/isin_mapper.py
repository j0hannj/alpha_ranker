"""
ISIN mapping utilities for the research platform.

This module sits on top of `core.isin` and exposes a clean, reusable API:

- map_isin_to_ticker(isin) -> ticker or None
- map_ticker_to_isin(ticker) -> isin or None (best-effort)
- resolve_asset_identifier(raw_id) -> dict with ticker/isin/exchange

The goal is to let higher-level components (universe builder, data pipeline,
portfolio optimizer) work with ISINs as the primary identifier, while still
supporting legacy ticker-based workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    from core.isin import resolve_isin as _resolve_isin_core
except Exception:  # pragma: no cover - defensive
    _resolve_isin_core = None  # type: ignore


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
    Attempt to infer ISIN from a ticker.

    NOTE: In general this mapping is not invertible; this is a best-effort
    helper that inspects the existing ISIN cache maintained by `core.isin`.
    """
    from pathlib import Path
    import json

    from core.isin import CACHE_FILE  # reuse existing cache path

    try:
        if Path(CACHE_FILE).exists():
            data = json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
            for isin, info in data.items():
                if info.get("ticker") == ticker:
                    return isin
    except Exception:
        return None
    return None


def resolve_asset_identifier(raw: str) -> Optional[AssetIdentifier]:
    """
    Resolve a user-provided identifier (ticker or ISIN) into a canonical form.

    - If it looks like an ISIN (12 chars, alpha prefix), try ISIN→ticker.
    - Otherwise treat as ticker and try to recover an ISIN from cache.
    """
    raw = raw.strip().upper()
    if not raw:
        return None

    # ISIN pattern: 2 letters + 10 alphanumerics
    if len(raw) == 12 and raw[:2].isalpha():
        ticker = map_isin_to_ticker(raw) or raw
        return AssetIdentifier(ticker=ticker, isin=raw, source="isin")

    # Otherwise assume ticker, attempt to backfill ISIN
    isin = map_ticker_to_isin(raw)
    return AssetIdentifier(ticker=raw, isin=isin, source="ticker")

