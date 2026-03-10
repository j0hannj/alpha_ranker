"""
Multi-source ISIN resolution chain: ticker -> ISIN.

Priority: local_cache (SQLite) -> static_isins.csv -> FMP (if key) -> yfinance.
OpenFIGI was removed (it returns FIGI only, not ISIN). Persistent cache via core.api_cache.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Exchange suffix mapping: Yahoo Finance format -> (replacement, OpenFIGI exchCode)
EXCHANGE_SUFFIXES = {
    ".PA": ("", "FP"),   # Euronext Paris
    ".AS": ("", "NA"),   # Euronext Amsterdam
    ".BR": ("", "EB"),   # Euronext Brussels
    ".DE": ("", "GY"),   # XETRA
    ".L":  ("", "LN"),   # London
    ".MI": ("", "IM"),   # Borsa Italiana
    ".MC": ("", "SM"),   # BME Madrid
    ".SW": ("", "SE"),   # SIX Swiss
    ".TO": ("", "CT"),   # Toronto
    ".HK": ("", "HK"),   # Hong Kong
    ".T":  ("", "JP"),   # Tokyo
    ".AX": ("", "AU"),   # Australia
}

STATIC_ISINS_PATH = Path(__file__).parent.parent.parent / "db" / "static_isins.csv"
# OpenFIGI /v3/mapping returns FIGI only, not ISIN — removed from chain (was dead)
RESOLUTION_CHAIN = ["local_cache", "static_file", "fmp", "yahoo_finance"]


def normalize_ticker(yahoo_ticker: str) -> Tuple[str, Optional[str]]:
    """
    Returns (clean_ticker, exchange_code) from a Yahoo Finance ticker.
    E.g. "SAN.PA" -> ("SAN", "FP"), "AAPL" -> ("AAPL", "US").
    """
    if not yahoo_ticker or not isinstance(yahoo_ticker, str):
        return (yahoo_ticker or "", None)
    t = yahoo_ticker.strip().upper()
    for suffix, (replacement, exch_code) in EXCHANGE_SUFFIXES.items():
        if t.endswith(suffix):
            clean = t[: -len(suffix)] + replacement
            return (clean, exch_code)
    return (t, "US")


def _load_static_isins() -> Dict[str, str]:
    """Load ticker->ISIN from db/static_isins.csv. Returns dict."""
    out = {}
    if not STATIC_ISINS_PATH.exists():
        logger.warning("Static ISIN file missing: %s — add ticker,isin,exchange,name for better coverage", STATIC_ISINS_PATH)
        return out
    try:
        with open(STATIC_ISINS_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return out
        header = [c.strip().lower() for c in lines[0].split(",")]
        ticker_col = None
        isin_col = None
        for i, h in enumerate(header):
            if h in ("ticker", "symbol"):
                ticker_col = i
            if h == "isin":
                isin_col = i
        if ticker_col is None or isin_col is None:
            return out
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > max(ticker_col, isin_col):
                ticker, isin = parts[ticker_col], parts[isin_col]
                if ticker and isin and len(isin) >= 10:
                    out[ticker.upper()] = isin.upper()
                    # Also store normalized (no suffix) if applicable
                    clean, _ = normalize_ticker(ticker)
                    if clean != ticker.upper():
                        out[clean] = isin.upper()
    except Exception as e:
        logger.warning("Failed to load static ISINs: %s", e)
    return out


_static_cache: Optional[Dict[str, str]] = None


def _get_static_map() -> Dict[str, str]:
    global _static_cache
    if _static_cache is None:
        _static_cache = _load_static_isins()
    return _static_cache


def _resolve_from_local_cache(ticker: str, clean: str) -> Optional[str]:
    try:
        from core.api_cache import get_isin_map
        m = get_isin_map()
        return m.get(ticker) or m.get(clean)
    except Exception:
        return None


def _resolve_from_static(ticker: str, clean: str) -> Optional[str]:
    m = _get_static_map()
    return m.get(ticker.upper()) or m.get(clean.upper())


def _resolve_from_fmp(ticker: str, clean: str) -> Optional[str]:
    """Resolve ISIN via FMP API (search or profile). Returns ISIN if FMP_API_KEY is set."""
    import os
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import urllib.request
        # FMP /v3/search returns isin in results; /v3/profile also has isin
        query = (clean or ticker).strip()
        url = f"https://financialmodelingprep.com/api/v3/search?query={query}&limit=5&apikey={api_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "AlphaRanker/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if not data or not isinstance(data, list):
            return None
        qnorm = (query or "").replace(".", "-").strip().upper()
        for item in data:
            sym = (item.get("symbol") or "").strip().upper().replace(".", "-")
            if sym != qnorm:
                continue
            isin = (item.get("isin") or item.get("ISIN") or "").strip()
            if isin and len(isin) >= 10:
                return isin.upper()
        # Fallback: profile endpoint often has isin
        url2 = f"https://financialmodelingprep.com/api/v3/profile/{query}?apikey={api_key}"
        req2 = urllib.request.Request(url2, headers={"User-Agent": "AlphaRanker/1.0"})
        with urllib.request.urlopen(req2, timeout=10) as r2:
            data2 = json.loads(r2.read().decode())
        if data2 and isinstance(data2, list) and data2:
            isin = (data2[0].get("isin") or data2[0].get("ISIN") or "").strip()
            if isin and len(isin) >= 10:
                return isin.upper()
    except Exception as e:
        logger.debug("FMP ISIN lookup failed for %s: %s", ticker, e)
    return None


def _resolve_from_yfinance(ticker: str) -> Optional[str]:
    """Resolve ISIN via yfinance. Prefer .isin property (correct); fallback to .info."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # Method 1: .isin property (the correct way — not in .info)
        try:
            isin_val = getattr(t, "isin", None)
            if isin_val and isinstance(isin_val, str):
                isin = isin_val.strip()
                if len(isin) >= 10 and isin != "-":
                    return isin.upper()
        except Exception:
            pass
        # Method 2: fallback via .info (rarely present)
        try:
            info = getattr(t, "info", None) or {}
            isin = (info.get("isin") or info.get("ISIN") or "").strip()
            if isin and len(isin) >= 10 and isin != "-":
                return isin.upper()
        except Exception:
            pass
    except Exception:
        pass
    return None


def resolve_ticker_to_isin(ticker: str, use_chain: bool = True) -> Optional[str]:
    """
    Resolve a single ticker to ISIN using the multi-source chain.
    Persists result to local cache when found.
    """
    if not ticker or not isinstance(ticker, str):
        return None
    clean, exchange = normalize_ticker(ticker)
    t = ticker.strip().upper()
    clean_u = clean.upper() if clean else t

    if use_chain:
        # 1. Local cache (SQLite isin_map)
        isin = _resolve_from_local_cache(ticker, clean_u)
        if isin:
            return isin

        # 2. Static file
        isin = _resolve_from_static(ticker, clean_u)
        if isin:
            _persist_isin(ticker, isin)
            return isin

        # 3. FMP API (returns ISIN if FMP_API_KEY set)
        isin = _resolve_from_fmp(ticker, clean_u)
        if isin:
            _persist_isin(ticker, isin)
            return isin

        # 4. Yahoo Finance
        isin = _resolve_from_yfinance(ticker)
        if isin:
            _persist_isin(ticker, isin)
            return isin

    return None


def _persist_isin(ticker: str, isin: str) -> None:
    try:
        from core.api_cache import get_isin_map, set_isin_map
        m = dict(get_isin_map())
        m[ticker.strip().upper()] = isin.strip().upper()
        set_isin_map(m)
    except Exception:
        pass


def resolve_ticker_to_isin_with_source(ticker: str) -> Tuple[Optional[str], str]:
    """Returns (isin, source_name). source_name is one of RESOLUTION_CHAIN or 'missing'."""
    if not ticker or not isinstance(ticker, str):
        return (None, "missing")
    clean, exchange = normalize_ticker(ticker)
    t = ticker.strip().upper()
    clean_u = clean.upper() if clean else t

    isin = _resolve_from_local_cache(ticker, clean_u)
    if isin:
        return (isin, "local_cache")

    isin = _resolve_from_static(ticker, clean_u)
    if isin:
        _persist_isin(ticker, isin)
        return (isin, "static_file")

    isin = _resolve_from_fmp(ticker, clean_u)
    if isin:
        _persist_isin(ticker, isin)
        return (isin, "fmp")

    isin = _resolve_from_yfinance(ticker)
    if isin:
        _persist_isin(ticker, isin)
        return (isin, "yahoo_finance")

    return (None, "missing")


def batch_resolve_tickers_to_isin(
    tickers: List[str],
    callback=None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Resolve many tickers to ISINs. Returns (ticker_to_isin, source_counts).
    Uses chain for each; reports coverage.
    """
    result: Dict[str, str] = {}
    source_counts: Dict[str, int] = {s: 0 for s in RESOLUTION_CHAIN}
    source_counts["missing"] = 0

    for i, t in enumerate(tickers):
        isin, source = resolve_ticker_to_isin_with_source(t)
        if isin:
            result[t] = isin
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts[source] = 1
        else:
            source_counts["missing"] += 1
        if callback and (i + 1) % 100 == 0:
            callback(f"ISIN resolution: {i + 1}/{len(tickers)}")

    return result, source_counts


def report_isin_coverage(total: int, resolved: int, source_stats: Dict[str, int]) -> None:
    """Log coverage and warn if low."""
    pct = (resolved / total * 100) if total > 0 else 0.0
    logger.info("ISIN coverage: %d/%d (%.1f%%)", resolved, total, pct)
    for source, count in source_stats.items():
        if count > 0:
            logger.info("  - %s: %d", source, count)
    if pct < 80 and total >= 50:
        logger.warning("LOW ISIN COVERAGE (%.1f%%) — model quality may be degraded", pct)
