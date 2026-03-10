"""
Data Layer — Institutional Grade
==================================
Single source of truth for all financial data.
Each function exists exactly once. All data is timestamped.

Sources:
  - Yahoo Finance → prices, basic info
  - FMP → fundamentals, ratios, universe
  - FRED → macro indicators
  - exchangerate.host → FX rates
"""
import calendar
import logging
import os, json, urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env before any API calls
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    logger.debug("dotenv not installed, relying on env vars")

CACHE_DIR = Path(__file__).parent.parent.parent / "db"
CACHE_DIR.mkdir(exist_ok=True)

# ── CONFIG (sans hardcoding) ───────────────────────────────────
def _get_data_config():
    """Start year, max years, min tickers depuis la config."""
    try:
        from .engine_config import get_data_settings
        cfg = get_data_settings()
        start = int(cfg.get("data_start_year", 2011))
        max_y = int(cfg.get("data_max_history_years", 15))
        min_t = int(cfg.get("data_min_tickers", 2500))
        return start, max_y, min_t
    except Exception:
        return 2011, 15, 2500


UNIVERSE_CACHE = CACHE_DIR / "universe_cache.json"


def _read_html_with_headers(url: str):
    """
    pd.read_html avec un vrai User-Agent pour éviter les 403 (Wikipedia, etc.).
    """
    import io

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="replace")
    return pd.read_html(io.StringIO(html))


WIKIPEDIA_INDICES = [
    # US — large / mid / small / broad
    {"name": "S&P 500", "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "suffix": "", "fix_dots": True},
    {"name": "S&P 400 MidCap", "url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "suffix": "", "fix_dots": True},
    {"name": "S&P 600 SmallCap", "url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "suffix": "", "fix_dots": True},
    {"name": "S&P 100", "url": "https://en.wikipedia.org/wiki/S%26P_100", "suffix": "", "fix_dots": True},
    {"name": "Nasdaq 100", "url": "https://en.wikipedia.org/wiki/Nasdaq-100", "suffix": "", "fix_dots": True},
    {"name": "Dow Jones Industrial Average", "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "suffix": "", "fix_dots": True},
    {"name": "Dow Jones Transportation Average", "url": "https://en.wikipedia.org/wiki/Dow_Jones_Transportation_Average", "suffix": "", "fix_dots": True},
    {"name": "Dow Jones Utility Average", "url": "https://en.wikipedia.org/wiki/Dow_Jones_Utility_Average", "suffix": "", "fix_dots": True},

    # Canada
    {"name": "S&P/TSX 60", "url": "https://en.wikipedia.org/wiki/S%26P/TSX_60", "suffix": ".TO", "fix_dots": False},
    {"name": "S&P/TSX Composite", "url": "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index", "suffix": ".TO", "fix_dots": False},

    # UK
    {"name": "FTSE 100", "url": "https://en.wikipedia.org/wiki/FTSE_100_Index", "suffix": ".L", "fix_dots": False},
    {"name": "FTSE 250", "url": "https://en.wikipedia.org/wiki/FTSE_250_Index", "suffix": ".L", "fix_dots": False},

    # Germany
    {"name": "DAX 40", "url": "https://en.wikipedia.org/wiki/DAX", "suffix": ".DE", "fix_dots": False},
    {"name": "MDAX", "url": "https://en.wikipedia.org/wiki/MDAX", "suffix": ".DE", "fix_dots": False},
    {"name": "SDAX", "url": "https://en.wikipedia.org/wiki/SDAX", "suffix": ".DE", "fix_dots": False},
    {"name": "TecDAX", "url": "https://en.wikipedia.org/wiki/TecDAX", "suffix": ".DE", "fix_dots": False},

    # France
    {"name": "CAC 40", "url": "https://en.wikipedia.org/wiki/CAC_40", "suffix": ".PA", "fix_dots": False},
    {"name": "SBF 120", "url": "https://en.wikipedia.org/wiki/SBF_120", "suffix": ".PA", "fix_dots": False},
    {"name": "CAC Next 20", "url": "https://en.wikipedia.org/wiki/CAC_Next_20", "suffix": ".PA", "fix_dots": False},
    {"name": "CAC Mid 60", "url": "https://en.wikipedia.org/wiki/CAC_Mid_60", "suffix": ".PA", "fix_dots": False},

    # Pan-Europe / Eurozone
    {"name": "Euro Stoxx 50", "url": "https://en.wikipedia.org/wiki/EURO_STOXX_50", "suffix": "", "fix_dots": False},
    {"name": "STOXX Europe 600", "url": "https://en.wikipedia.org/wiki/STOXX_Europe_600", "suffix": "", "fix_dots": False},

    # Benelux / Suisse
    {"name": "AEX 25", "url": "https://en.wikipedia.org/wiki/AEX_index", "suffix": ".AS", "fix_dots": False},
    {"name": "BEL 20", "url": "https://en.wikipedia.org/wiki/BEL_20", "suffix": ".BR", "fix_dots": False},
    {"name": "SMI", "url": "https://en.wikipedia.org/wiki/Swiss_Market_Index", "suffix": ".SW", "fix_dots": False},
    {"name": "SLI", "url": "https://en.wikipedia.org/wiki/Swiss_Leader_Index", "suffix": ".SW", "fix_dots": False},
    {"name": "SPI", "url": "https://en.wikipedia.org/wiki/Swiss_Performance_Index", "suffix": ".SW", "fix_dots": False},

    # Southern Europe
    {"name": "IBEX 35", "url": "https://en.wikipedia.org/wiki/IBEX_35", "suffix": ".MC", "fix_dots": False},
    {"name": "FTSE MIB", "url": "https://en.wikipedia.org/wiki/FTSE_MIB", "suffix": ".MI", "fix_dots": False},
    {"name": "PSI 20", "url": "https://en.wikipedia.org/wiki/PSI_20", "suffix": ".LS", "fix_dots": False},
    {"name": "ATX", "url": "https://en.wikipedia.org/wiki/Austrian_Traded_Index", "suffix": ".VI", "fix_dots": False},

    # Nordics
    {"name": "OBX 25", "url": "https://en.wikipedia.org/wiki/OBX_Index", "suffix": ".OL", "fix_dots": False},
    {"name": "OMX Stockholm 30", "url": "https://en.wikipedia.org/wiki/OMX_Stockholm_30", "suffix": ".ST", "fix_dots": False},
    {"name": "OMX Helsinki 25", "url": "https://en.wikipedia.org/wiki/OMX_Helsinki_25", "suffix": ".HE", "fix_dots": False},
    {"name": "OMX Copenhagen 25", "url": "https://en.wikipedia.org/wiki/OMX_Copenhagen_25", "suffix": ".CO", "fix_dots": False},

    # Ireland / CEE
    {"name": "ISEQ 20", "url": "https://en.wikipedia.org/wiki/ISEQ_20", "suffix": ".IR", "fix_dots": False},
    {"name": "WIG 20", "url": "https://en.wikipedia.org/wiki/WIG20", "suffix": ".WA", "fix_dots": False},

    # Japan / Asia-Pacific developed
    {"name": "Nikkei 225", "url": "https://en.wikipedia.org/wiki/Nikkei_225", "suffix": ".T", "fix_dots": False},
    {"name": "S&P/ASX 200", "url": "https://en.wikipedia.org/wiki/S%26P/ASX_200", "suffix": ".AX", "fix_dots": False},
    {"name": "Hang Seng Index", "url": "https://en.wikipedia.org/wiki/Hang_Seng_Index", "suffix": ".HK", "fix_dots": False},
    {"name": "STI", "url": "https://en.wikipedia.org/wiki/Straits_Times_Index", "suffix": ".SI", "fix_dots": False},
    {"name": "S&P/NZX 50", "url": "https://en.wikipedia.org/wiki/S%26P/NZX_50_Index", "suffix": ".NZ", "fix_dots": False},

    # India
    {"name": "BSE SENSEX", "url": "https://en.wikipedia.org/wiki/BSE_SENSEX", "suffix": ".BO", "fix_dots": False},
    {"name": "NIFTY 50", "url": "https://en.wikipedia.org/wiki/NIFTY_50", "suffix": ".NS", "fix_dots": False},

    # Emerging / others (sélection d’indices avec tableau)
    {"name": "TA-35", "url": "https://en.wikipedia.org/wiki/TA-35_Index", "suffix": ".TA", "fix_dots": False},
    {"name": "Ibovespa", "url": "https://en.wikipedia.org/wiki/Ibovespa", "suffix": ".SA", "fix_dots": False},
    {"name": "FTSE/JSE Top 40", "url": "https://en.wikipedia.org/wiki/FTSE/JSE_Top_40_Index", "suffix": ".JO", "fix_dots": False},
    {"name": "BIST 30", "url": "https://en.wikipedia.org/wiki/BIST_30", "suffix": ".IS", "fix_dots": False},
]


def _discover_from_wikipedia(callback=None):
    """Scrape Wikipedia pour les constituants de plusieurs grands indices mondiaux."""
    discovered: dict[str, dict] = {}
    for idx in WIKIPEDIA_INDICES:
        try:
            if callback:
                callback(f"Wikipedia: {idx['name']}...")
            tables = _read_html_with_headers(idx["url"])
            ticker_col = None
            target_table = None
            for t in tables:
                for col in t.columns:
                    col_str = str(col).lower()
                    if any(x in col_str for x in ["ticker", "symbol", "epic", "code", "stock"]):
                        sample = t[col].dropna().astype(str).head(5).tolist()
                        if sample and all(len(s.strip()) < 15 for s in sample):
                            ticker_col = col
                            target_table = t
                            break
                if ticker_col:
                    break
            if target_table is None:
                logger.warning("Wikipedia %s: no ticker column found", idx["name"])
                if callback:
                    callback(f"  {idx['name']}: no ticker column found, skipping")
                continue
            count = 0
            for raw in target_table[ticker_col].dropna().astype(str):
                tck = raw.strip()
                if not tck or len(tck) > 15:
                    continue
                if tck.lower() in ("ticker", "symbol", "code", "epic", "stock"):
                    continue
                if idx.get("fix_dots"):
                    tck = tck.replace(".", "-")
                if idx.get("suffix") and "." not in tck:
                    tck = tck + idx["suffix"]
                if tck not in discovered:
                    discovered[tck] = {"source": f"wikipedia_{idx['name']}"}
                    count += 1
            if callback:
                callback(f"  {idx['name']}: {count} tickers")
        except Exception as e:
            logger.warning("Wikipedia %s failed: %s", idx["name"], e)
            if callback:
                callback(f"  {idx['name']} ERROR: {e}")
    if callback:
        callback(f"Wikipedia total: {len(discovered)} unique tickers")
    return discovered


def _discover_from_yf_search(callback=None):
    """Découverte thématique via yfinance.Search."""
    import yfinance as yf

    discovered: dict[str, dict] = {}
    queries = [
        "technology stocks large cap",
        "healthcare stocks large cap",
        "financial stocks large cap",
        "energy stocks large cap",
        "consumer stocks large cap",
        "industrial stocks large cap",
        "semiconductor stocks",
        "biotech stocks",
        "software stocks",
        "renewable energy stocks",
        "AI artificial intelligence stocks",
        "cloud computing stocks",
        "cybersecurity stocks",
        "electric vehicle stocks",
        "European large cap stocks",
        "UK large cap stocks",
        "German stocks DAX",
        "French stocks CAC",
        "Swiss stocks",
    ]
    for query in queries:
        try:
            res = yf.Search(query)
            quotes = getattr(res, "quotes", None)
            if not quotes:
                continue
            if not isinstance(quotes, list):
                quotes = list(quotes)
            count = 0
            for q in quotes[:30]:
                sym = q.get("symbol") if isinstance(q, dict) else getattr(q, "symbol", None)
                if sym and isinstance(sym, str) and sym not in discovered:
                    discovered[sym] = {
                        "shortName": q.get("shortname") or q.get("longname") if isinstance(q, dict) else None,
                        "exchange": q.get("exchange") if isinstance(q, dict) else None,
                        "source": "yf_search",
                    }
                    count += 1
            if callback and count:
                callback(f"  Search '{query}': +{count}")
        except Exception as e:
            logger.warning("yfinance Search '%s' failed: %s", query, e)
            if callback:
                callback(f"  Search error for '{query}': {e}")
    if callback:
        callback(f"  yf.Search total: {len(discovered)} tickers")
    return discovered


def _enrich_with_yfinance(tickers_to_enrich, callback=None):
    """
    Enrichir sector + marketCap via yfinance.

    IMPORTANT: Ne pas appeler ça sur des milliers de tickers en une fois,
    sinon on se fait rate-limiter. Limiter en amont dans scan_and_expand_universe.
    """
    import yfinance as yf
    import time as _time

    enriched: dict[str, dict] = {}
    batch_size = 50

    for i in range(0, len(tickers_to_enrich), batch_size):
        batch = tickers_to_enrich[i : i + batch_size]
        if callback and i % 100 == 0:
            callback(f"  Enriching {i}/{len(tickers_to_enrich)} via yfinance...")
        for sym in batch:
            try:
                info = yf.Ticker(sym).info
                if info and info.get("marketCap"):
                    enriched[sym] = {
                        "shortName": info.get("shortName"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        "marketCap": info.get("marketCap"),
                        "currentPrice": info.get("currentPrice")
                        or info.get("regularMarketPrice"),
                        "country": info.get("country"),
                        "exchange": info.get("exchange"),
                    }
            except Exception as e:
                logger.debug("yfinance info for %s failed: %s", sym, e)
        # petites pauses entre batchs pour éviter le rate limit Yahoo
        if i + batch_size < len(tickers_to_enrich):
            try:
                _time.sleep(1)
            except Exception:
                pass

    if callback:
        callback(f"  Enriched via yfinance: {len(enriched)}/{len(tickers_to_enrich)}")
    return enriched


def _discover_from_fmp_search(callback=None):
    """Découverte via FMP /api/v3/search (gratuit) – actuellement souvent 403.

    NOTE: n'est plus appelée par scan_and_expand_universe. Laissons-la en
    place pour des usages futurs éventuels, avec fail-fast sur 403.
    """
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        return {}
    discovered: dict[str, dict] = {}
    searches = [
        "technology",
        "healthcare",
        "financial",
        "energy",
        "consumer",
        "industrial",
        "semiconductor",
        "software",
        "pharma",
        "telecom",
        "mining",
        "luxury",
    ]
    fmp_is_dead = False
    for term in searches:
        if fmp_is_dead:
            break
        for exchange in ["NASDAQ", "NYSE", "EURONEXT", "XETRA", "LSE"]:
            try:
                url = (
                    "https://financialmodelingprep.com/api/v3/search"
                    f"?query={term}&limit=50&exchange={exchange}&apikey={api_key}"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read().decode())
                for item in data or []:
                    sym = item.get("symbol")
                    if sym and sym not in discovered:
                        discovered[sym] = {
                            "shortName": item.get("name"),
                            "exchange": item.get("exchangeShortName"),
                            "source": "fmp_search",
                        }
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    logger.warning("FMP search returned 403 — stopping all FMP search calls")
                    if callback:
                        callback("FMP API not available (403). Skipping FMP search.")
                    fmp_is_dead = True
                    break
                logger.warning("FMP search '%s' on %s failed: %s", term, exchange, e)
                if callback:
                    callback(f"  FMP search '{term}' {exchange} error: {e}")
            except Exception as e:
                logger.warning("FMP search '%s' on %s failed: %s", term, exchange, e)
                if callback:
                    callback(f"  FMP search '{term}' {exchange} error: {e}")
    if callback:
        callback(f"  FMP search: {len(discovered)} tickers")
    return discovered


def scan_and_expand_universe(callback=None):
    """
    Découvrir des actions en combinant uniquement les sources qui marchent actuellement:
    - Wikipedia (indices majeurs)
    - yfinance.Search (requêtes thématiques)
    - FMP /api/v3/search (gratuit, si clé dispo)
    """
    logger.info("scan_and_expand_universe: start (Wikipedia + yf.Search + FMP search)")
    try:
        from . import portfolio
        from .engine_config import get_universe_settings
    except Exception as e:
        logger.warning("scan_and_expand_universe imports: %s", e)
        return _load_known_universe_as_fundamentals()

    uv = get_universe_settings()
    logger.info("scan_and_expand_universe: universe_settings=%s", uv)

    min_cap_cfg = uv.get("fmp_min_market_cap")
    min_cap = int(min_cap_cfg) if isinstance(min_cap_cfg, (int, float)) else 0

    # Cache: si scan récent et univers assez gros, réutiliser
    scan_freq_h = uv.get("scan_frequency_hours", 24)
    if scan_freq_h and scan_freq_h > 0:
        try:
            last = portfolio.get_setting("universe_last_scan")
            if last:
                from datetime import datetime as dt
                last_dt = dt.fromisoformat(last)
                age_sec = (datetime.now() - last_dt).total_seconds()
                if age_sec < scan_freq_h * 3600:
                    logger.info(
                        "scan_and_expand_universe: using cached scan (last=%s, age=%.1fh, freq_h=%s)",
                        last,
                        age_sec / 3600.0,
                        scan_freq_h,
                    )
                    if callback:
                        callback("Universe: using cached scan (recent).")
                    known = portfolio.get_universe() or {}
                    today = datetime.now().strftime("%Y-%m-%d")
                    active = {
                        t: {
                            "shortName": info.get("shortName") or t,
                            "sector": info.get("sector"),
                            "industry": info.get("industry"),
                            "marketCap": info.get("marketCap"),
                            "currentPrice": info.get("currentPrice"),
                            "country": info.get("country"),
                            "exchange": info.get("exchange"),
                            "date": today,
                            "discovered_at": info.get("discovered_at"),
                        }
                        for t, info in known.items()
                        if (info.get("marketCap") or 0) >= min_cap
                    }
                    min_cached = int(uv.get("min_cached_universe_size", 1000))
                    if len(active) >= min_cached:
                        logger.info(
                            "scan_and_expand_universe: returning cached DB universe of %d active stocks",
                            len(active),
                        )
                        return active
                    logger.info(
                        "scan_and_expand_universe: cached universe too small (active=%d, known=%d, min_cached=%d), forcing full rescan",
                        len(active),
                        len(known),
                        min_cached,
                    )
        except Exception as e:
            logger.debug("scan_and_expand_universe cache check failed: %s", e)

    discovered: dict[str, dict] = {}

    # 1) Wikipedia
    wiki = _discover_from_wikipedia(callback)
    for k, v in wiki.items():
        discovered.setdefault(k, {}).update(v)
    if callback:
        callback(f"After Wikipedia: {len(discovered)} tickers")

    # 2) yfinance.Search
    yf_search = _discover_from_yf_search(callback)
    for k, v in yf_search.items():
        if k not in discovered:
            discovered[k] = v
    if callback:
        callback(f"After yf.Search: {len(discovered)} tickers")

    # 3) (FMP search désactivé par défaut, API retourne 403)
    #    On n'appelle plus _discover_from_fmp_search() ici.
    if callback:
        callback(f"Total discovered (raw): {len(discovered)} unique tickers")
    logger.info("scan_and_expand_universe: total discovered (raw)=%d", len(discovered))

    # 4) Merge avec univers connu, enrichir ensuite via yfinance (FMP profile → 403)
    known = portfolio.get_universe() or {}
    known_set = set(known.keys())
    new_tickers = set(discovered.keys()) - known_set
    if callback:
        callback(f"Known: {len(known_set)} | New discoveries: {len(new_tickers)}")

    # Enrichir uniquement les tickers nouveaux ou incomplets via yfinance (limités)
    need_enrich = [
        t
        for t in new_tickers
        if not known.get(t, {}).get("marketCap")
    ]
    max_enrich = int(uv.get("max_profile_enrichment", 200))
    if need_enrich:
        if callback:
            callback(f"Enriching {min(len(need_enrich), max_enrich)} stocks via yfinance...")
        profiles = _enrich_with_yfinance(need_enrich[:max_enrich], callback)
    else:
        profiles = {}

    now_iso = datetime.now().isoformat()
    today = datetime.now().strftime("%Y-%m-%d")
    full_universe: dict[str, dict] = dict(known)
    for sym, base_info in discovered.items():
        base = full_universe.get(sym, {})
        prof = profiles.get(sym) or {}
        full_universe[sym] = {
            "shortName": prof.get("shortName") or base_info.get("shortName") or base.get("shortName") or sym,
            "sector": prof.get("sector") or base_info.get("sector") or base.get("sector"),
            "industry": prof.get("industry") or base_info.get("industry") or base.get("industry"),
            "marketCap": prof.get("marketCap") or base_info.get("marketCap") or base.get("marketCap"),
            "currentPrice": prof.get("currentPrice") or base_info.get("currentPrice") or base.get("currentPrice"),
            "country": base_info.get("country") or base.get("country"),
            "exchange": base_info.get("exchange") or base.get("exchange"),
            "date": today,
            "discovered_at": base.get("discovered_at") or (now_iso if sym in new_tickers else None),
            "last_seen_at": now_iso,
        }

    logger.info("scan_and_expand_universe: saving universe of %d tickers to DB", len(full_universe))
    portfolio.save_universe(full_universe)
    try:
        portfolio.set_setting("universe_last_scan", now_iso)
    except Exception as e:
        logger.warning("scan_and_expand_universe: failed to persist universe_last_scan: %s", e)
    logger.info("scan_and_expand_universe: end")

    # 5) Univers ACTIF filtré par market cap
    active = {
        t: {
            "shortName": info.get("shortName") or t,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "currentPrice": info.get("currentPrice"),
            "country": info.get("country"),
            "exchange": info.get("exchange"),
            "date": today,
            "discovered_at": info.get("discovered_at"),
        }
        for t, info in full_universe.items()
        if (info.get("marketCap") or 0) >= min_cap
    }
    logger.info("scan_and_expand_universe: active universe size=%d (min_mcap=%s)", len(active), min_cap)
    if callback:
        human_cap = f"{min_cap/1e9:.1f}B" if min_cap >= 1e9 else f"{min_cap/1e6:.1f}M"
        callback(f"Active universe: {len(active)} stocks (mcap >= {human_cap})")
    return active


def _fetch_all_traded(api_key: str, uv: dict, callback=None):
    """
    /api/v3/available-traded/list — GRATUIT.
    Retourne tous les instruments tradés; on filtre ensuite par exchange + type.
    """
    url = f"https://financialmodelingprep.com/api/v3/available-traded/list?apikey={api_key}"
    logger.info("scan_and_expand_universe: calling available-traded/list")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            all_instruments = json.loads(r.read().decode())
    except Exception as e:
        logger.warning("scan_and_expand_universe: available-traded/list failed: %s", e)
        if callback:
            callback(f"FMP available-traded/list error: {e}")
        return []

    if callback:
        callback(f"FMP returned {len(all_instruments)} instruments")

    # Exchanges wanted: from config, sinon set par défaut raisonnable
    cfg_exchanges = uv.get("fmp_exchanges") or []
    if cfg_exchanges:
        wanted = [e.lower() for e in cfg_exchanges]
    else:
        wanted = [
            "nyse",
            "nasdaq",
            "amex",
            "euronext",
            "xetra",
            "lse",
            "paris",
            "amsterdam",
            "brussels",
            "milan",
            "frankfurt",
        ]

    stocks = []
    for s in all_instruments or []:
        sym = (s.get("symbol") or "").strip()
        if not sym:
            continue
        exch = (s.get("exchangeShortName") or s.get("exchange") or "").strip()
        typ = (s.get("type") or "").lower()

        # Virer warrants, units, preferred, crypto, forex, fonds
        if any(x in sym for x in ["-W", "-U", ".WS", "-WT", "-R"]):
            continue
        if typ in ("crypto", "forex", "fund", "trust", "etf"):
            continue

        if wanted and not any(w in exch.lower() for w in wanted):
            continue

        stocks.append(s)

    if callback:
        callback(f"After exchange filter: {len(stocks)} stocks")
    logger.info("scan_and_expand_universe: after exchange/type filter -> %d instruments", len(stocks))
    return stocks


def _enrich_with_profiles(tickers, api_key: str, callback=None):
    """
    /api/v3/profile/{sym1,sym2,...} — GRATUIT.
    Retourne sector, marketCap, PE, country... par batch (50).
    """
    fundamentals: dict[str, dict] = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            symbols = ",".join(batch)
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbols}?apikey={api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            for item in data or []:
                sym = item.get("symbol")
                if not sym:
                    continue
                fundamentals[sym] = {
                    "shortName": item.get("companyName"),
                    "sector": item.get("sector"),
                    "industry": item.get("industry"),
                    "marketCap": item.get("marketCap") or item.get("mktCap"),
                    "currentPrice": item.get("price"),
                    "beta": item.get("beta"),
                    "trailingPE": item.get("peRatio"),
                    "country": item.get("country"),
                    "exchange": item.get("exchangeShortName"),
                }
            if callback and (i // batch_size) % 4 == 0:
                callback(f"  Profiles: {min(i+batch_size, len(tickers))}/{len(tickers)}")
        except Exception as e:
            logger.warning("scan_and_expand_universe: profile batch starting at %d failed: %s", i, e)
            if callback:
                callback(f"  Profile error at {i}: {e}")
    logger.info("scan_and_expand_universe: enriched profiles for %d tickers", len(fundamentals))
    return fundamentals


def _load_known_universe_as_fundamentals():
    """Load known universe from DB and return as fundamentals-style dict."""
    try:
        from . import portfolio
        known = portfolio.get_universe()
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            t: {
                "shortName": info.get("shortName") or t,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "marketCap": info.get("marketCap"),
                "currentPrice": info.get("currentPrice"),
                "date": today,
                "discovered_at": info.get("discovered_at"),
            }
            for t, info in known.items()
        }
    except Exception as e:
        logger.warning("_load_known_universe_as_fundamentals: %s", e)
        return {}


def _fetch_universe_yfinance(callback=None):
    """
    Fallback universe construction when no FMP key is available.
    ETF list and search queries come from universe_settings (no hardcoded lists in code).
    Returns (tickers, fundamentals_dict).
    """
    import yfinance as yf

    try:
        from .engine_config import get_universe_settings
        uv = get_universe_settings()
    except Exception:
        uv = {}
    etf_list = uv.get("yf_etf_tickers") or []
    search_list = uv.get("yf_search_queries") or []

    tickers: set[str] = set()
    fundamentals: dict[str, dict] = {}

    for etf in etf_list:
        try:
            if callback:
                callback(f"Scanning {etf} holdings...")
            t = yf.Ticker(etf)
            holdings = None
            try:
                # Modern yfinance exposes funds_data.top_holdings for ETFs
                holdings = getattr(getattr(t, "funds_data", None), "top_holdings", None)
            except Exception as e:
                logger.debug("yfinance holdings for %s failed: %s", etf, e)
            if holdings is not None and not holdings.empty:
                syms = [s for s in holdings.index.tolist() if isinstance(s, str) and len(s) < 12]
                tickers.update(syms)
                if callback:
                    callback(f"  {etf}: {len(syms)} holdings")
        except Exception as e:
            logger.warning("ETF %s holdings failed: %s", etf, e)
            if callback:
                callback(f"  {etf}: {e}")

    # Étendre l'univers via yfinance.Search (multi-queries, plus de résultats)
    for query in search_list:
        try:
            results = yf.Search(query)
            quotes = getattr(results, "quotes", None)
            if quotes:
                if not isinstance(quotes, list):
                    quotes = list(quotes)
                # Prendre plus de résultats par requête pour gonfler l'univers
                for q in quotes[:80]:
                    sym = q.get("symbol") if isinstance(q, dict) else getattr(q, "symbol", None)
                    if sym and isinstance(sym, str):
                        tickers.add(sym)
        except Exception as e:
            logger.warning("yfinance Search '%s' failed: %s", query, e)
            if callback:
                callback(f"  Search error: {e}")

    # Si malgré tout l'univers est petit (<300), essayer d'utiliser yfinance.Screener si dispo
    if len(tickers) < 300 and getattr(yf, "Screener", None):
        try:
            if callback:
                callback("Using yfinance Screener for extra large caps...")
            screener = yf.Screener()
            body = {
                "offset": 0,
                "size": 500,
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
            screener.set_default_body(body)
            result = getattr(screener, "response", None) or {}
            quotes = result.get("quotes", []) if isinstance(result, dict) else []
            extra = 0
            for q in quotes:
                sym = (q.get("symbol") or q.get("ticker") or "").strip()
                if sym:
                    tickers.add(sym)
                    extra += 1
            if callback:
                callback(f"  Screener added ~{extra} tickers")
            logger.info("_fetch_universe_yfinance: Screener added %d tickers (total=%d)", extra, len(tickers))
        except Exception as e:
            logger.debug("_fetch_universe_yfinance: Screener fallback failed: %s", e)

    # 3) Fundamentals via yfinance.info for the discovered universe
    if callback:
        callback(f"Fetching info for {len(tickers)} stocks (yfinance)...")
    today = datetime.now().strftime("%Y-%m-%d")
    for t in list(tickers):
        try:
            info = yf.Ticker(t).info
            if info and info.get("marketCap"):
                keys = [
                    "marketCap", "trailingPE", "forwardPE", "sector", "industry",
                    "shortName", "currentPrice", "beta", "dividendYield",
                    "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
                ]
                fundamentals[t] = {"date": today, **{k: info.get(k) for k in keys}}
        except Exception as e:
            logger.debug("yfinance info for %s failed: %s", t, e)

    if callback:
        callback(f"Universe (yfinance): {len(tickers)} stocks")
    return sorted(tickers), fundamentals


def fetch_universe_cached(years=5, callback=None):
    """
    Cached universe (tickers + fundamentals), refreshed at most every 24h.
    Source: scan_and_expand_universe, which only uses FMP free endpoints
    (available-traded/list + profile) and yfinance fallback when FMP is blocked.
    """
    api_key = os.environ.get("FMP_API_KEY")

    if UNIVERSE_CACHE.exists():
        try:
            cache = json.loads(UNIVERSE_CACHE.read_text(encoding="utf-8"))
            date_str = cache.get("date")
            cached_tickers = cache.get("tickers") or []
            cached_fund = cache.get("fundamentals") or {}
            if date_str:
                age_sec = (datetime.now() - datetime.fromisoformat(date_str)).total_seconds()
            else:
                age_sec = 1e9
            if age_sec < 86400 and len(cached_tickers) > 100:
                if callback:
                    callback(
                        f"Universe cache: {len(cached_tickers)} stocks "
                        f"({int(age_sec // 3600)}h old)"
                    )
                return cached_tickers, cached_fund
        except Exception as e:
            logger.warning("Universe cache read failed: %s", e)
            if callback:
                callback(f"Cache error: {e}")

    if not api_key:
        # Sans clé FMP, on laisse scan_and_expand_universe gérer la partie yfinance-only.
        logger.warning("fetch_universe_cached: missing FMP_API_KEY, using scan_and_expand_universe without FMP")

    # Utiliser le pipeline moderne qui n'appelle que des endpoints free ou yfinance.
    fundamentals = scan_and_expand_universe(callback=callback) or {}
    tickers = sorted(fundamentals.keys())

    try:
        UNIVERSE_CACHE.write_text(
            json.dumps(
                {
                    "tickers": tickers,
                    "fundamentals": fundamentals,
                    "date": datetime.now().isoformat(),
                    "source": "scan_and_expand_universe",
                },
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Universe cache write failed: %s", e)
        if callback:
            callback(f"Cache write error: {e}")

    return tickers, fundamentals


def _dataframe_from_cache_dict(cached):
    """Reconstruct DataFrame from cache dict (orient='split')."""
    if not cached or "data" not in cached:
        return None
    try:
        idx = pd.to_datetime(cached["index"]) if isinstance(cached["index"], list) else cached["index"]
        return pd.DataFrame(cached["data"], index=idx, columns=cached.get("columns"))
    except Exception:
        return None


def fetch_universe(years=5, callback=None):
    """
    Build global equity universe.
    PRIMARY source: scan_and_expand_universe (FMP free endpoints + yfinance fallback).

    For backward compatibility, returns (tickers, prices, fundamentals)
    even though prices are now usually downloaded in fetch_all_data.
    """
    import yfinance as yf

    # Utiliser le même univers que le pipeline principal (scan_and_expand_universe),
    # éventuellement mis en cache via fetch_universe_cached.
    tickers, fundamentals = fetch_universe_cached(years=years, callback=callback)
    if callback:
        callback(f"Universe: {len(tickers)} stocks")

    # Download prices for requested history window
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    if callback:
        callback(f"Downloading prices for {len(tickers)} stocks...")
    prices = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )

    return tickers, prices, fundamentals

# ── SINGLE PRICE ──────────────────────────────────────────────
def fetch_price(ticker):
    """Fetch current price for a single ticker. Returns dict with timestamp."""
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        return {"ticker": ticker, "price": round(float(p),2) if p else None,
                "currency": info.get("currency","USD"), "name": info.get("shortName",ticker),
                "sector": info.get("sector"), "change_pct": info.get("regularMarketChangePercent"),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
    except:
        return {"ticker": ticker, "price": None, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}

# ── BATCH PRICES ──────────────────────────────────────────────
def fetch_prices(tickers):
    """Fetch current prices for multiple tickers. All timestamped."""
    prices = {}
    for t in tickers:
        r = fetch_price(t)
        if r["price"] is not None:
            prices[t] = r
    return prices

# ── ISIN (FMP profile / bulk, objectif 2500–4000+) ───────────────
def get_largest_tickers_by_market_cap(n=4000, callback=None):
    """
    Retourne les n plus gros tickers par market cap (cache yahoo_info).
    Si pas assez en cache, complète avec l'univers stocké.
    """
    from .api_cache import get_tickers_by_market_cap, get_stored_universe_list
    by_cap = get_tickers_by_market_cap()
    tickers = [t for t, _ in by_cap[:n]]
    if len(tickers) < n:
        stored = get_stored_universe_list()
        for t in stored:
            if t not in tickers:
                tickers.append(t)
                if len(tickers) >= n:
                    break
    if callback:
        callback(f"Data: {len(tickers)} plus grosses cap pour ISIN (objectif {n})")
    return tickers[:n]


def fetch_large_cap_isins(target=4000, callback=None):
    """
    Récupère au moins target ISIN (2500 ou 4000) pour les plus grosses capitalisations.
    Peut être lancé par l'agent IA [ACTION:fetch_large_cap_isins:4000].
    """
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        if callback:
            callback("Clé FMP requise pour récupérer les ISIN (Settings).")
        return {}
    tickers = get_largest_tickers_by_market_cap(n=target, callback=callback)
    return fetch_isins_fmp(tickers, api_key, callback=callback, min_isins=target)


def fetch_isins_fmp(tickers, api_key, callback=None, chunk_size=5, min_isins=None):
    """
    Récupère au moins min_isins (ou data_min_isins) ISIN via FMP.
    Essaie d'abord le bulk (profile-bulk?part=N), sinon profile un par un.
    Met à jour le cache isin_map (ticker -> isin). Sauvegarde incrémentale.
    """
    from .api_cache import get_isin_map, set_isin_map
    if not api_key or not tickers:
        return {}
    if min_isins is not None:
        target = max(1000, int(min_isins))
    else:
        try:
            from .engine_config import get_data_settings
            target = max(1000, int(get_data_settings().get("data_min_isins", 2500)))
        except Exception:
            target = 2500
    out = dict(get_isin_map())
    target = max(target, len(out))

    # 1) Essai bulk (paginated): beaucoup d'ISIN en peu d'appels
    for part in range(20):
        try:
            url = f"https://financialmodelingprep.com/api/v4/profile/bulk?part={part}&apikey={api_key}"
            with urllib.request.urlopen(url, timeout=45) as r:
                data = json.loads(r.read().decode())
            if not data:
                break
            for item in (data or []):
                sym = (item.get("symbol") or "").strip()
                isin = (item.get("isin") or "").strip()
                if sym and isin and len(isin) >= 10:
                    out[sym] = isin
            if callback:
                callback(f"Data: ISIN bulk part {part} → {len(out)}")
            if len(out) >= target:
                set_isin_map(out)
                return out
        except Exception:
            break

    # 2) Fallback: profile un par un (ou petit batch si l'API l'accepte) pour les tickers manquants
    need = [t for t in tickers if t not in out]
    if len(out) >= target:
        set_isin_map(out)
        return out
    for i in range(0, min(len(need), max(target - len(out), 2500)), chunk_size):
        chunk = need[i : i + chunk_size]
        for sym in chunk:
            try:
                url = f"https://financialmodelingprep.com/api/v3/profile/{sym}?apikey={api_key}"
                with urllib.request.urlopen(url, timeout=15) as r:
                    data = json.loads(r.read().decode())
                for item in (data or []):
                    s = (item.get("symbol") or "").strip()
                    isin = (item.get("isin") or "").strip()
                    if s and isin and len(isin) >= 10:
                        out[s] = isin
            except Exception:
                pass
        if (i + chunk_size) % 200 < chunk_size and callback:
            callback(f"Data: ISIN {len(out)} (objectif {target})")
        if len(out) >= target:
            break
        try:
            import time
            time.sleep(0.05)
        except Exception:
            pass
    set_isin_map(out)
    return out


# ── FUNDAMENTALS (FMP) ────────────────────────────────────────
def fetch_fundamentals(ticker, api_key=None):
    """Fetch quarterly income statement from FMP. Timestamped by filing date."""
    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key: return []
    try:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&limit=40&apikey={api_key}"
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode())
        return [{"ticker":ticker, "date":item.get("date"), "filing_date":item.get("fillingDate") or item.get("filingDate") or item.get("date"),
                 "revenue":item.get("revenue"), "net_income":item.get("netIncome"),
                 "eps":item.get("eps"), "ebitda":item.get("ebitda"),
                 "gross_profit":item.get("grossProfit"), "operating_income":item.get("operatingIncome")}
                for item in data if item.get("date")]
    except: return []

# ── RATIOS (FMP) ──────────────────────────────────────────────
def fetch_ratios(ticker, api_key=None):
    """Fetch quarterly key metrics + ratios from FMP. Timestamped."""
    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key: return []
    results = []
    for endpoint in ["key-metrics","ratios"]:
        try:
            url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=quarter&limit=40&apikey={api_key}"
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode())
            for item in data:
                d = item.get("date")
                if not d: continue
                results.append({"ticker":ticker, "date":d, "source":endpoint, **{
                    k:item.get(k) for k in ["peRatio","pbRatio","enterpriseValueOverEBITDA",
                    "roe","returnOnTangibleAssets","debtToEquity","currentRatio",
                    "freeCashFlowPerShare","marketCap","dividendYield","payoutRatio",
                    "revenuePerShare","bookValuePerShare","grossProfitMargin",
                    "operatingProfitMargin","netProfitMargin",                    "priceEarningsToGrowthRatio",
                    "quickRatio"] if k in item}})
        except Exception as e:
            logger.warning("fetch_fmp_quarterly: request failed for %s: %s", ticker, e)
    return results

# ── MACRO (FRED) ──────────────────────────────────────────────
def fetch_macro(callback=None):
    """Fetch macro indicators from FRED. Cached in SQLite 24h."""
    try:
        from .api_cache import get, set
        cached = get("fred", "macro", max_age_hours=24)
        if cached is not None:
            if callback: callback("Macro: cache hit")
            return cached
    except Exception as e:
        logger.debug("fetch_macro: api_cache get failed: %s", e)
    if callback: callback("Macro: fetching FRED...")
    fallback = {"date":datetime.now().strftime("%Y-%m-%d"),
                "fed_funds_rate":4.5,"us_10y_yield":4.1,"yield_curve_slope":-0.4,
                "cpi_yoy":3.0,"oil_price":80,"vix":22,"credit_spread":1.5}
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key: return fallback
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        end = datetime.now(); start = end - timedelta(days=365)
        macro = {"date": end.strftime("%Y-%m-%d")}
        for name, sid in {"fed_funds_rate":"FEDFUNDS","us_10y_yield":"DGS10","us_2y_yield":"DGS2",
                          "vix":"VIXCLS","oil_price":"DCOILWTICO","credit_spread":"BAA10Y"}.items():
            try:
                s = fred.get_series(sid, start, end)
                if len(s) > 0: macro[name] = round(float(s.dropna().iloc[-1]),2)
            except Exception as e:
                logger.debug("fetch_macro: FRED series %s failed: %s", sid, e)
        if "us_10y_yield" in macro and "us_2y_yield" in macro:
            macro["yield_curve_slope"] = round(macro["us_10y_yield"]-macro["us_2y_yield"],2)
        try:
            set("fred", "macro", macro)
        except Exception as e:
            logger.debug("fetch_macro: api_cache set failed: %s", e)
        return macro
    except Exception as e:
        logger.warning("fetch_macro: FRED fetch failed, using fallback: %s", e)
        return fallback

# ── FX (exchangerate.host + yfinance fallback) ────────────────
def fetch_fx(base="EUR", targets=None, callback=None):
    """Fetch FX rates. Cached in SQLite 24h."""
    if targets is None: targets = ["USD","GBP","CHF"]
    cache_key = f"{base}_{'_'.join(sorted(targets))}"
    try:
        from .api_cache import get, set
        cached = get("fx", cache_key, max_age_hours=24)
        if cached is not None:
            if callback: callback("FX: cache hit")
            return cached
    except Exception as e:
        logger.debug("fetch_fx: api_cache get failed: %s", e)
    if callback: callback("FX: fetching rates...")
    rates = {"date": datetime.now().strftime("%Y-%m-%d"), "base": base}
    try:
        url = f"https://api.exchangerate.host/latest?base={base}&symbols={','.join(targets)}"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        if data.get("success") or data.get("rates"):
            for t in targets:
                if t in data.get("rates",{}):
                    rates[f"{base}_{t}"] = round(data["rates"][t],4)
    except Exception as e:
        logger.warning("fetch_fx: exchangerate.host request failed: %s", e)
    if not any(f"{base}_{t}" in rates for t in targets):
        try:
            import yfinance as yf
            for t in targets:
                pair = f"{base}{t}=X"
                p = yf.Ticker(pair).info.get("regularMarketPrice")
                if p: rates[f"{base}_{t}"] = round(float(p),4)
        except Exception as e:
            logger.warning("fetch_fx: yfinance fallback failed: %s", e)
    if f"{base}_USD" not in rates: rates[f"{base}_USD"] = 1.08
    if f"{base}_GBP" not in rates: rates[f"{base}_GBP"] = 0.86
    try:
        set("fx", cache_key, rates)
    except Exception as e:
        logger.debug("fetch_fx: api_cache set failed: %s", e)
    return rates

# ── COMBINED FETCH ────────────────────────────────────────────
def fetch_all_data(tickers=None, years=5, callback=None):
    """Fetch all data sources in one call. Scans markets to discover new stocks, then downloads prices.
    Used by run_full_pipeline to get everything needed."""
    try:
        from .api_cache import clear_older_than_days
        clear_older_than_days(30)
    except Exception as e:
        logger.debug("fetch_all_data: clear_older_than_days failed: %s", e)

    api_key = os.environ.get("FMP_API_KEY")
    try:
        from .engine_config import get_universe_settings
        uv = get_universe_settings()
    except Exception:
        uv = {}

    if api_key and uv.get("auto_scan", True):
        if callback:
            callback("Scanning markets for new stocks...")
        fundamentals = scan_and_expand_universe(callback=callback)
        universe_tickers = list(fundamentals.keys())
    elif api_key:
        if callback:
            callback("Universe: loading from DB (auto-scan off)...")
        fundamentals = _load_known_universe_as_fundamentals()
        universe_tickers = list(fundamentals.keys())
    else:
        if callback:
            callback("FMP key required for universe. Set it in Settings.")
        universe_tickers, fundamentals = fetch_universe_cached(years, callback=callback)

    if tickers:
        universe_tickers = sorted(set(universe_tickers + list(tickers)))
    # max_universe_size: si défini explicitement dans les settings, on respecte.
    # Sinon, aucune limite artificielle côté code (seule limite = mémoire/temps yfinance).
    max_size = uv.get("max_universe_size")
    if isinstance(max_size, int) and max_size > 0 and len(universe_tickers) > max_size:
        universe_tickers = universe_tickers[:max_size]
        fundamentals = {t: fundamentals[t] for t in universe_tickers if t in fundamentals}

    # Prices for the (possibly extended) universe
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    if callback:
        callback(f"Data: downloading prices for {len(universe_tickers)} stocks...")
    prices = yf.download(
        universe_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )

    if callback: callback("Data: macro...")
    macro = fetch_macro(callback=callback)

    if callback: callback("Data: FX...")
    fx = fetch_fx(callback=callback)

    if callback: callback("Data: sentiment...")
    sentiment = {}
    try:
        from .news import batch_sentiment
        sentiment = batch_sentiment(list(fundamentals.keys())[:150], callback)
    except Exception as e:
        logger.warning("fetch_all_data: batch_sentiment failed: %s", e)

    if callback:
        n_t, n_f = len(universe_tickers), len(fundamentals)
        callback(f"Data done: {n_t} tickers | {n_f} fundamentals | sentiment {len(sentiment)}")

    now = datetime.now()
    now_iso = now.isoformat()
    now_display = now.strftime("%Y-%m-%d %H:%M")
    data_freshness = {
        "prices": {"last_update_timestamp": now_iso, "data_source": "Yahoo Finance", "display": now_display},
        "fundamentals": {
            "last_update_timestamp": now_iso,
            "data_source": "FMP" if os.environ.get("FMP_API_KEY") else "Yahoo Finance",
            "display": now_display,
        },
        "macro": {"last_update_timestamp": now_iso, "data_source": "FRED", "display": now_display},
    }
    return {
        "tickers": universe_tickers,
        "prices": prices,
        "fundamentals": fundamentals,
        "macro": macro,
        "fx": fx,
        "sentiment": sentiment,
        "fetched_at": now_iso,
        "data_freshness": data_freshness,
    }
