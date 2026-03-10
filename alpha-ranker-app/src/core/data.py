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


# ── UNIVERSE (FMP screener + DB persistence, découverte automatique) ────────
UNIVERSE_CACHE = CACHE_DIR / "universe_cache.json"


def scan_and_expand_universe(callback=None):
    """
    Scan markets via FMP screener to discover new stocks.
    Merge with known universe from DB and persist. Called at start of fetch_all_data.
    Returns dict ticker -> {shortName, sector, marketCap, currentPrice, ...} (fundamentals format).
    """
    api_key = os.environ.get("FMP_API_KEY")
    try:
        from . import portfolio
        from .engine_config import get_universe_settings
    except Exception as e:
        logger.warning("scan_and_expand_universe imports: %s", e)
        return _load_known_universe_as_fundamentals()

    # Optional: skip scan if last scan was recent
    uv = get_universe_settings()
    scan_freq_h = uv.get("scan_frequency_hours", 24)
    if scan_freq_h and scan_freq_h > 0:
        try:
            last = portfolio.get_setting("universe_last_scan")
            if last:
                from datetime import datetime as dt
                last_dt = dt.fromisoformat(last)
                if (datetime.now() - last_dt).total_seconds() < scan_freq_h * 3600:
                    if callback:
                        callback("Universe: using cached scan (recent).")
                    return _load_known_universe_as_fundamentals()
        except Exception:
            pass

    if not api_key:
        if callback:
            callback("FMP key needed to discover new stocks. Set it in Settings.")
        return _load_known_universe_as_fundamentals()

    exchange_list = uv.get("fmp_exchanges") or []
    min_cap = uv.get("fmp_min_market_cap") or 500_000_000
    limit = uv.get("fmp_screener_limit") or 3000
    today = datetime.now().strftime("%Y-%m-%d")

    discovered = {}
    for exchange_str in exchange_list:
        label = exchange_str.split(",")[0] if exchange_str else "?"
        try:
            if callback:
                callback(f"Scanning {label}...")
            url = (
                "https://financialmodelingprep.com/api/v3/stock-screener"
                f"?marketCapMoreThan={int(min_cap)}"
                f"&isActivelyTrading=true"
                f"&exchange={exchange_str}"
                f"&limit={int(limit)}"
                f"&apikey={api_key}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            for item in data or []:
                sym = item.get("symbol")
                if not sym:
                    continue
                discovered[sym] = {
                    "name": item.get("companyName"),
                    "sector": item.get("sector"),
                    "industry": item.get("industry"),
                    "marketCap": item.get("marketCap") or item.get("mktCap"),
                    "price": item.get("price"),
                    "country": item.get("country"),
                    "exchange": item.get("exchangeShortName") or label,
                }
            if callback:
                callback(f"  {label}: {len(data) if isinstance(data, list) else 0} stocks found")
        except Exception as e:
            logger.warning("FMP screener %s failed: %s", exchange_str, e)
            if callback:
                callback(f"  {label} scan error: {e}")

    known = portfolio.get_universe()
    known_set = set(known.keys())
    new_tickers = set(discovered.keys()) - known_set
    if callback:
        callback(f"Scan complete: {len(discovered)} total, {len(new_tickers)} NEW discoveries")

    # Merge: known + discovered (discovered overwrites for fresh data)
    full = dict(known)
    for t, info in discovered.items():
        full[t] = {
            "shortName": info.get("name") or t,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "currentPrice": info.get("price"),
            "country": info.get("country"),
            "exchange": info.get("exchange"),
            "date": today,
        }

    portfolio.save_universe(full)
    try:
        portfolio.set_setting("universe_last_scan", datetime.now().isoformat())
    except Exception:
        pass
    return full


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
                "currentPrice": None,
                "date": today,
            }
            for t, info in known.items()
        }
    except Exception as e:
        logger.warning("_load_known_universe_as_fundamentals: %s", e)
        return {}


def _fetch_universe_fmp(api_key, callback=None):
    """
    FMP stock screener → 1000–2000+ global equities.
    Primary source when FMP_API_KEY is configured.
    Exchanges, min market cap and limit come from universe_settings (no hardcoded lists).
    Returns (tickers, fundamentals_dict).
    """
    try:
        from .engine_config import get_universe_settings
        uv = get_universe_settings()
    except Exception:
        uv = {}
    exchange_list = uv.get("fmp_exchanges") or []
    min_cap = uv.get("fmp_min_market_cap") or 500_000_000
    limit = uv.get("fmp_screener_limit") or 2000

    tickers: list[str] = []
    fundamentals: dict[str, dict] = {}

    for exchange_str in exchange_list:
        label = exchange_str.split(",")[0] if exchange_str else "?"
        try:
            if callback:
                callback(f"FMP screener: {label}...")
            url = (
                "https://financialmodelingprep.com/api/v3/stock-screener"
                f"?marketCapMoreThan={int(min_cap)}"
                f"&isActivelyTrading=true"
                f"&exchange={exchange_str}"
                f"&limit={int(limit)}"
                f"&apikey={api_key}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())

            count = 0
            today = datetime.now().strftime("%Y-%m-%d")
            for item in data or []:
                sym = item.get("symbol")
                if not sym:
                    continue
                tickers.append(sym)
                count += 1
                fundamentals[sym] = {
                    "date": today,
                    "marketCap": item.get("marketCap") or item.get("mktCap"),
                    "sector": item.get("sector"),
                    "industry": item.get("industry"),
                    "shortName": item.get("companyName"),
                    "currentPrice": item.get("price"),
                    "beta": item.get("beta"),
                    "trailingPE": item.get("peRatio") if item.get("peRatio") else None,
                    "forwardPE": item.get("peRatio"),
                    "dividendYield": item.get("lastAnnualDividend"),
                    "volume": item.get("volume"),
                    "exchange": item.get("exchangeShortName"),
                    "country": item.get("country"),
                }
            if callback:
                callback(f"  {label}: {count} stocks")
        except Exception as e:
            logger.warning("FMP screener %s failed: %s", exchange_str, e)
            if callback:
                callback(f"  {label} ERROR: {e}")

    tickers = sorted(set(tickers))
    return tickers, fundamentals


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

    for query in search_list:
        try:
            results = yf.Search(query)
            quotes = getattr(results, "quotes", None)
            if quotes:
                if not isinstance(quotes, list):
                    quotes = list(quotes)
                for q in quotes[:30]:
                    sym = q.get("symbol") if isinstance(q, dict) else getattr(q, "symbol", None)
                    if sym and isinstance(sym, str):
                        tickers.add(sym)
        except Exception as e:
            logger.warning("yfinance Search '%s' failed: %s", query, e)
            if callback:
                callback(f"  Search error: {e}")

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
    Primary source: FMP screener. If FMP_API_KEY is missing, no universe is built.
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
        # Dans une vraie app financière, l'univers vient du data provider.
        # Ici: FMP est obligatoire pour construire l'univers, on échoue explicitement.
        if callback:
            callback("FMP_API_KEY is not set. Configure it in Settings to build the equity universe.")
        logger.warning("fetch_universe_cached: missing FMP_API_KEY, returning empty universe")
        return [], {}

    tickers, fundamentals = _fetch_universe_fmp(api_key, callback)

    try:
        UNIVERSE_CACHE.write_text(
            json.dumps(
                {
                    "tickers": tickers,
                    "fundamentals": fundamentals,
                    "date": datetime.now().isoformat(),
                    "source": "fmp",
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
    PRIMARY source: FMP stock screener (requires FMP_API_KEY).
    SECONDARY source: yfinance (no key, more limited).

    For backward compatibility, returns (tickers, prices, fundamentals)
    even though prices are now usually downloaded in fetch_all_data.
    """
    import yfinance as yf

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
    """Fetch all data sources in one call. Returns dict with timestamps.
    Used by run_full_pipeline to get everything needed."""
    try:
        from .api_cache import clear_older_than_days
        clear_older_than_days(30)
    except Exception as e:
        logger.debug("fetch_all_data: clear_older_than_days failed: %s", e)
    if callback:
        callback("Data: universe...")
    universe_tickers, fundamentals = fetch_universe_cached(years, callback=callback)
    if tickers:
        universe_tickers = sorted(set(universe_tickers + tickers))

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
