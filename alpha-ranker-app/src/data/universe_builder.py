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


# ═══════════════════════════════════════════════════════════════
# AUTO-FILL UNIVERSE (large caps until target count)
# ═══════════════════════════════════════════════════════════════


def fetch_largecap_candidates(
    region: str = "global",
    count: int = 800,
) -> List[Dict[str, str]]:
    """
    Fetch large-cap stock candidates from free sources (Wikipedia, FMP if key).
    Returns list of dicts: [{"ticker": "AAPL", "name": "Apple Inc", "sector": "Technology"}, ...].
    """
    candidates: List[Dict[str, str]] = []
    seen: set = set()

    # 1) S&P 500 from Wikipedia (US large caps)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        for _, row in df.iterrows():
            ticker = str(row.get("Symbol", row.get("Ticker", ""))).strip().replace(".", "-")
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            candidates.append({
                "ticker": ticker,
                "name": str(row.get("Security", row.get("Company", ticker)))[:60],
                "sector": str(row.get("GICS Sector", ""))[:40],
            })
            if len(candidates) >= count:
                return candidates
    except Exception:
        pass

    # 2) Nasdaq-100 from Wikipedia
    if len(candidates) < count:
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
            for t in tables:
                col = "Ticker" if "Ticker" in t.columns else "Symbol"
                if col not in t.columns:
                    continue
                for _, row in t.iterrows():
                    ticker = str(row[col]).strip().replace(".", "-")
                    if ticker and ticker not in seen:
                        seen.add(ticker)
                        candidates.append({
                            "ticker": ticker,
                            "name": str(row.get("Company", row.get("Security", ticker)))[:60],
                            "sector": "",
                        })
                    if len(candidates) >= count:
                        return candidates
                break
        except Exception:
            pass

    # 3) FMP stock screener by market cap (if API key)
    if len(candidates) < count:
        fmp_key = os.environ.get("FMP_API_KEY")
        if fmp_key:
            try:
                import urllib.request
                import json
                url = (
                    f"https://financialmodelingprep.com/api/v3/stock-screener?"
                    f"marketCapMoreThan=1000000000&limit={count + 200}&apikey={fmp_key}"
                )
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read().decode())
                for item in (data or []):
                    sym = (item.get("symbol") or "").strip()
                    if not sym or sym in seen:
                        continue
                    seen.add(sym)
                    candidates.append({
                        "ticker": sym,
                        "name": (item.get("companyName") or sym)[:60],
                        "sector": (item.get("sector") or "")[:40],
                    })
                    if len(candidates) >= count:
                        break
            except Exception:
                pass

    return candidates[:count]


def auto_fill_universe(
    target_count: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[object] = None,
) -> Dict:
    """
    Fill the universe with large-cap stocks (with valid ISIN) until target_count is reached.
    Uses stored universe + isin_map from core.api_cache. Adds tickers and persists ISINs.
    Returns:
        added: list of {"ticker", "isin", "name", "sector", "source": "auto_fill"}
        failed: list of tickers where ISIN resolution failed
        universe_size: total valid count after run
        target_reached: bool
    """
    try:
        from core.api_cache import (
            get_stored_universe_list,
            set_stored_universe_list,
            get_isin_map,
            set_isin_map,
        )
    except ImportError:
        return {
            "added": [],
            "failed": [],
            "universe_size": 0,
            "target_reached": False,
        }

    current_tickers = set(get_stored_universe_list())
    isin_map = get_isin_map()
    current_valid = sum(1 for t in current_tickers if isin_map.get(t))
    needed = target_count - current_valid

    if needed <= 0:
        return {
            "added": [],
            "failed": [],
            "universe_size": current_valid,
            "target_reached": True,
        }

    fetch_count = min(2000, int(needed * 1.5) + 50)
    candidates = fetch_largecap_candidates(region="global", count=fetch_count)
    candidates = [c for c in candidates if c["ticker"] not in current_tickers]

    added: List[Dict] = []
    failed: List[str] = []

    for i, candidate in enumerate(candidates):
        if getattr(cancel_event, "is_set", lambda: False)():
            break
        if len(added) >= needed:
            break

        ticker = candidate["ticker"]
        if progress_callback:
            progress_callback(
                len(added),
                needed,
                f"Résolution ISIN pour {ticker}…",
            )

        isin = isin_mapper.map_ticker_to_isin(ticker)

        if isin:
            entry = {
                "ticker": ticker,
                "isin": isin,
                "name": candidate.get("name", ""),
                "sector": candidate.get("sector", ""),
                "source": "auto_fill",
            }
            added.append(entry)
            current_tickers.add(ticker)
        else:
            failed.append(ticker)

    # Persist: extend stored universe and isin_map (already updated by mapper)
    if added:
        new_list = list(current_tickers)
        set_stored_universe_list(new_list)

    return {
        "added": added,
        "failed": failed,
        "universe_size": current_valid + len(added),
        "target_reached": (current_valid + len(added)) >= target_count,
    }

