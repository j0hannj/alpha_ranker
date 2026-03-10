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
import os, json, urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Load .env before any API calls
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed, rely on env vars

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
    Univers via get_universe_tickers (objectif 2500+). Prix année par année: 2011 → sauvegarde → 2012 → …
    Premier fetch long, ensuite on ne requête que l'année courante. Reprise possible (checkpoint par année).
    """
    import yfinance as yf
    from .api_cache import get as cache_get, set as cache_set

    start_year, max_history_years, _ = _get_data_config()
    end_year = datetime.now().year
    start_year = max(start_year, end_year - max_history_years + 1)
    year_list = list(range(start_year, end_year + 1))
    if not year_list:
        year_list = [end_year]

    if callback:
        try:
            from .api_cache import get_cache_status
            st = get_cache_status()
            path_short = str(Path(st["path"]).parent.name) + "/" + Path(st["path"]).name
            years_cached = [x["year"] for x in st["prices_yearly_years"]]
            n_tk = st.get("n_tickers")
            tick_str = f" | {n_tk} tickers" if n_tk is not None else ""
            if years_cached:
                callback(f"Cache: {path_short}{tick_str} | {len(years_cached)} années ({min(years_cached)}..{max(years_cached)})")
            else:
                callback(f"Cache: {path_short}{tick_str} | aucune année encore (premier run long)")
        except Exception:
            pass

    tickers = get_universe_tickers(callback=callback)
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

    # Prix année par année: cache par an, pas tout recommencer
    HOURS_20Y = 24 * 365 * 20
    parts = []
    for idx, y in enumerate(year_list):
        is_current = y == end_year
        max_age_h = 24 if is_current else HOURS_20Y
        cached = cache_get("prices_yearly", str(y), max_age_hours=max_age_h)
        if cached is not None:
            df = _dataframe_from_cache_dict(cached)
            if df is not None and not df.empty:
                if callback:
                    callback(f"Data: année {y} ({idx+1}/{len(year_list)}) cache")
                parts.append(df)
                continue
        if callback:
            callback(f"Data: téléchargement année {y} ({idx+1}/{len(year_list)})…")
        end_date = datetime.now().strftime("%Y-%m-%d") if is_current else f"{y}-12-31"
        try:
            year_prices = yf.download(
                tickers, start=f"{y}-01-01", end=end_date,
                group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
        except Exception:
            year_prices = pd.DataFrame()
        if year_prices.empty or (hasattr(year_prices, "columns") and len(year_prices.columns) == 0):
            continue
        try:
            cache_set("prices_yearly", str(y), year_prices.to_dict(orient="split"))
        except Exception:
            pass
        if callback:
            callback(f"Data: année {y} sauvegardée ({idx+1}/{len(year_list)})")
        parts.append(year_prices)

    if not parts:
        end = datetime.now()
        start = datetime(start_year, 1, 1)
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
                    "operatingProfitMargin","netProfitMargin","priceEarningsToGrowthRatio",
                    "quickRatio"] if k in item}})
        except: pass
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
    except Exception:
        pass
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
            except: pass
        if "us_10y_yield" in macro and "us_2y_yield" in macro:
            macro["yield_curve_slope"] = round(macro["us_10y_yield"]-macro["us_2y_yield"],2)
        try:
            set("fred", "macro", macro)
        except Exception:
            pass
        return macro
    except: return fallback

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
    except Exception:
        pass
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
    except: pass
    if not any(f"{base}_{t}" in rates for t in targets):
        try:
            import yfinance as yf
            for t in targets:
                pair = f"{base}{t}=X"
                p = yf.Ticker(pair).info.get("regularMarketPrice")
                if p: rates[f"{base}_{t}"] = round(float(p),4)
        except: pass
    if f"{base}_USD" not in rates: rates[f"{base}_USD"] = 1.08
    if f"{base}_GBP" not in rates: rates[f"{base}_GBP"] = 0.86
    try:
        set("fx", cache_key, rates)
    except Exception:
        pass
    return rates

# ── COMBINED FETCH ────────────────────────────────────────────
def fetch_all_data(tickers=None, years=5, callback=None):
    """Fetch all data sources in one call. Returns dict with timestamps.
    Used by run_full_pipeline to get everything needed."""
    try:
        from .api_cache import clear_older_than_days
        clear_older_than_days(30)
    except Exception:
        pass
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
    except: pass

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
