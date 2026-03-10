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

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd

from . import isin_mapper

logger = logging.getLogger(__name__)

# User-Agent for HTTP requests (Wikipedia blocks default Python/pandas)
WIKI_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_html(url: str) -> Optional[str]:
    """Fetch URL with a browser-like User-Agent to avoid 403 from Wikipedia."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("Fetch %s failed: %s", url[:50], e)
        return None


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


# ═══════════════════════════════════════════════════════════════
# AUTO-FILL UNIVERSE (chained sources, no cap)
# ═══════════════════════════════════════════════════════════════


def _build_source_chain(
    region: str,
) -> List[Tuple[str, Callable[[int], Iterator[List[Dict]]]]]:
    """Build ordered list of (source_name, generator_fn) based on region."""
    chain: List[Tuple[str, Callable[[int], Iterator[List[Dict]]]]] = []

    if region in ("us", "global"):
        chain.append(("S&P 500 (Wikipedia)", _source_wikipedia_sp500))
    if region in ("europe", "global"):
        chain.append(("STOXX 600 (Wikipedia)", _source_wikipedia_stoxx600))
    if region in ("europe", "global"):
        chain.append(("FTSE 100 (Wikipedia)", _source_wikipedia_ftse100))
    if region in ("asia", "global"):
        chain.append(("Nikkei 225 (Wikipedia)", _source_wikipedia_nikkei225))
    if region in ("us", "global"):
        chain.append(("yfinance screener US", lambda n: _source_yfinance_screener(n, market="us")))
    if region in ("europe", "global"):
        chain.append(("yfinance screener EU", lambda n: _source_yfinance_screener(n, market="europe")))
    if region in ("asia", "global"):
        chain.append(("yfinance screener Asia", lambda n: _source_yfinance_screener(n, market="asia")))
    chain.append(("Financial Modeling Prep", _source_fmp))
    return chain


def _source_wikipedia_sp500(needed: int) -> Iterator[List[Dict]]:
    """Deprecated: Wikipedia-based S&P 500 scraping removed. Yield empty."""
    logger.info("Auto-fill: S&P 500 via Wikipedia disabled")
    yield []


def _source_wikipedia_stoxx600(needed: int) -> Iterator[List[Dict]]:
    """Deprecated: Wikipedia-based STOXX 600 scraping removed. Yield empty."""
    logger.info("Auto-fill: STOXX 600 via Wikipedia disabled")
    yield []


def _source_wikipedia_ftse100(needed: int) -> Iterator[List[Dict]]:
    """Deprecated: Wikipedia-based FTSE 100 scraping removed. Yield empty."""
    logger.info("Auto-fill: FTSE 100 via Wikipedia disabled")
    yield []


def _source_wikipedia_nikkei225(needed: int) -> Iterator[List[Dict]]:
    """Deprecated: Wikipedia-based Nikkei 225 scraping removed. Yield empty."""
    logger.info("Auto-fill: Nikkei 225 via Wikipedia disabled")
    yield []


def _source_yfinance_screener(needed: int, market: str = "us") -> Iterator[List[Dict]]:
    """Yield large-cap stocks from yfinance screener if available (many versions lack Screener)."""
    try:
        import yfinance as yf
    except Exception:
        return
    if not getattr(yf, "Screener", None):
        logger.debug("yfinance.Screener not available in this version, skipping screener source")
        return
    page_size = 250
    offset = 0
    max_pages = 20
    for page in range(max_pages):
        try:
            screener = yf.Screener()
            body = {
                "offset": offset,
                "size": page_size,
                "sortField": "intradaymarketcap",
                "sortType": "desc",
                "quoteType": "equity",
                "query": {
                    "operator": "and",
                    "operands": [
                        {"operator": "gt", "operands": ["intradaymarketcap", 1_000_000_000]},
                    ],
                },
            }
            region_map = {"us": "us", "europe": "europe", "asia": "asia"}
            if market in region_map:
                body["query"]["operands"].append({"operator": "eq", "operands": ["region", region_map[market]]})
            screener.set_default_body(body)
            result = getattr(screener, "response", None) or {}
            quotes = result.get("quotes", []) if isinstance(result, dict) else []
            if not quotes:
                break
            batch = []
            for q in quotes:
                ticker = (q.get("symbol") or q.get("ticker") or "").strip()
                if ticker:
                    batch.append({
                        "ticker": ticker,
                        "name": (q.get("shortName") or q.get("longName") or ticker)[:60],
                        "marketCap": q.get("marketCap"),
                        "sector": (q.get("sector") or "")[:40],
                    })
            if batch:
                yield batch
            offset += page_size
            if len(quotes) < page_size:
                break
            time.sleep(0.5)
        except Exception as e:
            logger.warning("Auto-fill: yfinance screener %s page %s failed: %s", market, page, e)
            break


def _source_fmp(needed: int) -> Iterator[List[Dict]]:
    """Yield large-cap stocks from FMP API, paginated. Requires FMP_API_KEY."""
    if not os.getenv("FMP_API_KEY"):
        logger.info("FMP API key not configured, skipping")
        return
    page_size = 1000
    offset = 0
    max_pages = 10
    api_key = os.environ["FMP_API_KEY"]
    for page in range(max_pages):
        try:
            import urllib.request
            import json
            url = (
                f"https://financialmodelingprep.com/api/v3/stock-screener?"
                f"marketCapMoreThan=1000000000&limit={page_size}&offset={offset}&apikey={api_key}"
            )
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode())
            if not data:
                break
            batch = []
            for item in data:
                ticker = (item.get("symbol") or "").strip()
                if ticker:
                    batch.append({
                        "ticker": ticker,
                        "name": (item.get("companyName") or ticker)[:60],
                        "marketCap": item.get("marketCap"),
                        "sector": (item.get("sector") or "")[:40],
                    })
            if batch:
                yield batch
            offset += page_size
            if len(data) < page_size:
                break
            time.sleep(1.0)
        except Exception as e:
            logger.warning("Auto-fill: FMP page %s failed: %s", page, e)
            break


def fetch_largecap_candidates_chained(
    needed: int,
    existing_tickers: Set[str],
    existing_isins: Set[str],
    region: str = "global",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[object] = None,
) -> List[Dict]:
    """
    Fetch large-cap candidates from ALL sources in sequence until 'needed' are collected.
    Dedup by ticker and skip if ISIN already in universe. No artificial cap.
    """
    candidates: List[Dict] = []
    seen_tickers = set(existing_tickers)

    for source_name, source_fn in _build_source_chain(region):
        if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
            break
        if len(candidates) >= needed:
            break
        if progress_callback:
            progress_callback(len(candidates), needed, f"Source: {source_name}…")
        logger.info("Auto-fill: fetching from %s (still need %s)", source_name, needed - len(candidates))
        try:
            for batch in source_fn(needed - len(candidates)):
                if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                    break
                if len(candidates) >= needed:
                    break
                for stock in batch:
                    if len(candidates) >= needed:
                        break
                    ticker = (stock.get("ticker") or "").strip()
                    if not ticker or ticker in seen_tickers:
                        continue
                    seen_tickers.add(ticker)
                    stock = dict(stock)
                    stock["source"] = source_name
                    candidates.append(stock)
        except Exception as e:
            logger.warning("Auto-fill: source %s failed: %s", source_name, e)
    return candidates


def auto_fill_universe(
    target_count: int,
    current_universe: Optional[List[Dict]] = None,
    region: str = "global",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[object] = None,
) -> Dict:
    """
    Fill the universe with large-cap stocks until target_count. Chains ALL sources.
    Dedup by ticker and ISIN. Returns added, failed, universe_size, target_reached, sources_used.
    """
    try:
        from core.api_cache import (
            get_stored_universe_list,
            set_stored_universe_list,
            get_isin_map,
        )
    except ImportError:
        return {
            "added": [],
            "failed": [],
            "universe_size": 0,
            "target_reached": False,
            "sources_used": [],
        }

    tickers_list = get_stored_universe_list()
    isin_map = get_isin_map()
    existing_tickers: Set[str] = set(tickers_list)
    existing_isins: Set[str] = {v for v in isin_map.values() if v}
    if current_universe is not None:
        for s in current_universe:
            if s.get("ticker"):
                existing_tickers.add(s["ticker"])
            if s.get("isin"):
                existing_isins.add(s["isin"])
    current_valid = sum(1 for t in tickers_list if isin_map.get(t))
    needed = target_count - current_valid

    if needed <= 0:
        return {
            "added": [],
            "failed": [],
            "universe_size": current_valid,
            "target_reached": True,
            "sources_used": [],
        }

    overfetch = max(needed, int(needed * 1.5) + 100)
    candidates = fetch_largecap_candidates_chained(
        needed=overfetch,
        existing_tickers=existing_tickers,
        existing_isins=existing_isins,
        region=region,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    added: List[Dict] = []
    failed: List[str] = []
    sources_used: Set[str] = set()

    for candidate in candidates:
        if getattr(cancel_event, "is_set", lambda: False)():
            break
        if len(added) >= needed:
            break
        ticker = candidate["ticker"]
        if progress_callback:
            progress_callback(
                len(added),
                needed,
                f"Résolution ISIN pour {ticker} ({candidate.get('source', '?')})…",
            )
        isin = isin_mapper.map_ticker_to_isin(ticker)
        if isin and isin not in existing_isins:
            entry = {
                "ticker": ticker,
                "isin": isin,
                "name": candidate.get("name", ""),
                "sector": candidate.get("sector", ""),
                "source": candidate.get("source", "auto_fill"),
            }
            added.append(entry)
            existing_tickers.add(ticker)
            existing_isins.add(isin)
            sources_used.add(candidate.get("source", "unknown"))
        else:
            failed.append(ticker)

    if added:
        set_stored_universe_list(list(existing_tickers))

    return {
        "added": added,
        "failed": failed,
        "universe_size": current_valid + len(added),
        "target_reached": (current_valid + len(added)) >= target_count,
        "sources_used": list(sources_used),
    }

