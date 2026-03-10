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


# ── UNIVERSE (tous les equity disponibles, pas de liste fixe) ─────────────
def get_universe_tickers(callback=None):
    """
    Tous les symboles equity disponibles sur le marché. Pas de liste fixe.
    Avec FMP: appel API stock/list → tous les titres type "stock" (milliers).
    Sans FMP: fallback indices Wikipedia (S&P, Nasdaq, Russell). Liste minimale
    uniquement si tout échoue (< 50 symboles).
    """
    _, _, min_tickers = _get_data_config()
    tickers = []
    fmp_key = os.environ.get("FMP_API_KEY")
    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={fmp_key}"
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode())
            for item in (data or []):
                if (item.get("type") or "").lower() != "stock":
                    continue
                sym = (item.get("symbol") or "").strip()
                if not sym:
                    continue
                if "." in sym:
                    sym = sym.replace(".", "-")
                tickers.append(sym)
            tickers = list(dict.fromkeys(tickers))
            if callback:
                callback(f"Data: univers FMP — {len(tickers)} equity (tous disponibles, pas de liste fixe)")
            if tickers:
                return tickers
        except Exception:
            pass
    if callback:
        callback("Data: pas de clé FMP — fallback indices Wikipedia")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers.extend(tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            if "Ticker" in t.columns:
                tickers.extend(t["Ticker"].astype(str).str.replace(".", "-", regex=False).tolist())
                break
            if "Symbol" in t.columns:
                tickers.extend(t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
                break
    except Exception:
        pass
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index")
        for t in tables:
            if "Ticker" in t.columns:
                tickers.extend(t["Ticker"].astype(str).str.replace(".", "-", regex=False).tolist())
                break
            if "Symbol" in t.columns:
                tickers.extend(t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
                break
    except Exception:
        pass
    tickers = list(dict.fromkeys(tickers))
    if callback:
        callback(f"Data: univers indices — {len(tickers)} symboles")
    if len(tickers) < 50:
        if callback:
            callback("Data: liste minimale (fallback, pas de FMP)")
        tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V","UNH","XOM","JNJ","WMT","PG","MA","HD","CVX","MRK","ABBV","LLY","PEP","KO","COST","AVGO","TMO","MCD","ACN","CSCO","ABT","NEE","TXN","RTX","LOW","HON","AMGN","IBM","CAT","BA","GS","BLK","SPGI","AXP","DE","ISRG","REGN","VRTX","GILD","SYK","BKNG","CB","PLD","CI","CME","SHW","FCX","COP","EOG","SLB","HAL","LMT","GD","NOC","CRWD","PANW","PLTR","NET","DDOG","RKLB","HII"]
    return tickers


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
    Univers via get_universe_tickers (objectif 2500+). Prix mois par mois (cache et reprise par mois).
    """
    import yfinance as yf
    from .api_cache import get as cache_get, set as cache_set

    start_year, max_history_years, _ = _get_data_config()
    now = datetime.now()
    end_year, end_month = now.year, now.month
    start_year = max(start_year, end_year - max_history_years + 1)
    month_list = []
    for y in range(start_year, end_year + 1):
        m_start = 1 if y > start_year else 1
        m_end = end_month if y == end_year else 12
        for m in range(m_start, m_end + 1):
            month_list.append((y, m))
    if not month_list:
        month_list = [(end_year, end_month)]

    if callback:
        try:
            from .api_cache import get_cache_status
            st = get_cache_status()
            path_short = str(Path(st["path"]).parent.name) + "/" + Path(st["path"]).name
            periods = st.get("prices_monthly_periods", [])
            if isinstance(periods, list) and periods:
                p_min = min(x.get("period", x) for x in periods)
                p_max = max(x.get("period", x) for x in periods)
                callback(f"Cache: {path_short} | {len(periods)} mois ({p_min}..{p_max})")
            else:
                n_tk = st.get("n_tickers")
                tick_str = f" | {n_tk} tickers" if n_tk is not None else ""
                callback(f"Cache: {path_short}{tick_str} | premier run (mois par mois)")
        except Exception:
            pass

    # Liste stockée en DB = priorité : ce qui est en base reste même si l'API ne le renvoie plus
    from .api_cache import get_stored_universe_list, set_stored_universe_list
    stored = get_stored_universe_list()
    fresh = get_universe_tickers(callback=callback)
    tickers = list(set(stored) | set(fresh))
    if not tickers:
        tickers = fresh
    set_stored_universe_list(tickers)
    if callback and stored:
        callback(f"Data: univers {len(tickers)} (dont {len(stored)} déjà en base, conservés)")
    fmp_key = os.environ.get("FMP_API_KEY")
    if fmp_key and tickers and callback:
        try:
            fetch_isins_fmp(tickers, fmp_key, callback=callback)
        except Exception:
            pass
    fundamentals = {}
    for i, t in enumerate(tickers):
        cached_info = cache_get("yahoo_info", t, max_age_hours=24)
        if cached_info and isinstance(cached_info, dict):
            fundamentals[t] = cached_info
        else:
            try:
                info = yf.Ticker(t).info
                if info and info.get("marketCap"):
                    fundamentals[t] = {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        **{k: info.get(k) for k in [
                            "marketCap","trailingPE","forwardPE","pegRatio","priceToSalesTrailing12Months",
                            "priceToBook","enterpriseToEbitda","enterpriseToRevenue","profitMargins",
                            "operatingMargins","grossMargins","returnOnEquity","returnOnAssets",
                            "revenueGrowth","earningsGrowth","earningsQuarterlyGrowth","debtToEquity",
                            "currentRatio","quickRatio","freeCashflow","operatingCashflow","totalRevenue",
                            "totalDebt","totalCash","dividendYield","payoutRatio","beta","shortRatio",
                            "sharesOutstanding","heldPercentInstitutions","sector","industry","shortName",
                            "currentPrice","targetMeanPrice","recommendationKey","numberOfAnalystOpinions"
                        ]}
                    }
                    cache_set("yahoo_info", t, fundamentals[t])
            except Exception:
                pass
        if callback and (i + 1) % 50 == 0:
            callback(f"Data: {len(fundamentals)}/{len(tickers)} sauvegardés (reprise si arrêt)")
    if callback:
        callback(f"Data: {len(fundamentals)}/{len(tickers)} tickers sauvegardés")

    # Prix mois par mois: cache par mois. Si nouveaux tickers, on ne télécharge que les manquants puis fusion.
    HOURS_20Y = 24 * 365 * 20
    parts = []
    for idx, (y, m) in enumerate(month_list):
        is_current = (y == end_year and m == end_month)
        max_age_h = 24 if is_current else HOURS_20Y
        period_key = f"{y}-{m:02d}"
        cached = cache_get("prices_monthly", period_key, max_age_hours=max_age_h)
        df = None
        start_date = f"{y}-{m:02d}-01"
        last_day = calendar.monthrange(y, m)[1]
        end_date = f"{y}-{m:02d}-{last_day:02d}"
        if is_current:
            end_date = now.strftime("%Y-%m-%d")
        if cached is not None:
            df = _dataframe_from_cache_dict(cached)
            if df is not None and not df.empty:
                try:
                    cols = getattr(df, "columns", None)
                    if cols is not None and len(cols) > 0 and isinstance(cols[0], tuple):
                        tickers_in_cache = set(c[0] for c in cols if isinstance(c, tuple) and len(c) >= 2)
                    elif hasattr(cols, "get_level_values"):
                        tickers_in_cache = set(cols.get_level_values(0).unique())
                    elif hasattr(cols, "levels") and cols.levels:
                        tickers_in_cache = set(cols.levels[0])
                    else:
                        tickers_in_cache = set()
                    missing = [t for t in tickers if t not in tickers_in_cache]
                    if missing:
                        if callback and (idx + 1) % 12 == 0:
                            callback(f"Data: {period_key} ({idx+1}/{len(month_list)}) cache + {len(missing)} nouveaux")
                        try:
                            new_prices = yf.download(
                                missing, start=start_date, end=end_date,
                                group_by="ticker", auto_adjust=True, threads=True, progress=False
                            )
                            if not new_prices.empty and hasattr(new_prices, "columns") and len(new_prices.columns) > 0:
                                df = pd.concat([df, new_prices], axis=1)
                                try:
                                    cache_set("prices_monthly", period_key, df.to_dict(orient="split"))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    elif callback and (idx + 1) % 12 == 0:
                        callback(f"Data: {period_key} ({idx+1}/{len(month_list)}) cache")
                except Exception:
                    pass
        if df is None or df.empty:
            if callback and (idx + 1) % 12 == 1:
                callback(f"Data: téléchargement {period_key} ({idx+1}/{len(month_list)})…")
            try:
                month_prices = yf.download(
                    tickers, start=start_date, end=end_date,
                    group_by="ticker", auto_adjust=True, threads=True, progress=False
                )
            except Exception:
                month_prices = pd.DataFrame()
            if not month_prices.empty and hasattr(month_prices, "columns") and len(month_prices.columns) > 0:
                try:
                    cache_set("prices_monthly", period_key, month_prices.to_dict(orient="split"))
                except Exception:
                    pass
                if callback and (idx + 1) % 12 == 1:
                    callback(f"Data: {period_key} sauvegardé ({idx+1}/{len(month_list)})")
                parts.append(month_prices)
        else:
            parts.append(df)

    if not parts:
        end = now
        y0, m0 = month_list[0][0], month_list[0][1]
        start = datetime(y0, m0, 1)
        prices = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                            group_by="ticker", auto_adjust=True, threads=True)
    else:
        prices = pd.concat(parts, axis=0)
        if hasattr(prices.index, "duplicated"):
            prices = prices[~prices.index.duplicated(keep="first")]
        prices = prices.sort_index()
    if callback:
        callback(f"Data: universe prêt ({len(tickers)} tickers, {len(fundamentals)} infos)")
    try:
        cache_set("universe_meta", "count", {"n_tickers": len(tickers), "n_fundamentals": len(fundamentals), "updated": datetime.now().isoformat()})
    except Exception:
        pass
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
    if callback: callback("Data: universe...")
    universe_tickers, prices, fundamentals = fetch_universe(years, callback=callback)
    if tickers: universe_tickers = list(set(universe_tickers + tickers))

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

    now_iso = datetime.now().isoformat()
    now_display = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_freshness = {
        "prices": {"last_update_timestamp": now_iso, "data_source": "Yahoo Finance", "display": now_display},
        "fundamentals": {"last_update_timestamp": now_iso, "data_source": "Yahoo Finance", "display": now_display},
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
