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

# ── UNIVERSE ──────────────────────────────────────────────────
def fetch_universe(years=5):
    """Fetch S&P 500 universe + extras. Returns (tickers, prices_df, fundamentals_dict)."""
    import yfinance as yf
    tickers = []
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].str.replace(".","-",regex=False).tolist()
    except: pass
    if len(tickers) < 100:
        tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V","UNH",
                   "XOM","JNJ","WMT","PG","MA","HD","CVX","MRK","ABBV","LLY","PEP","KO","COST",
                   "AVGO","TMO","MCD","ACN","CSCO","ABT","NEE","TXN","RTX","LOW","HON","AMGN",
                   "IBM","CAT","BA","GS","BLK","SPGI","AXP","DE","ISRG","REGN","VRTX","GILD",
                   "SYK","BKNG","CB","PLD","CI","CME","SHW","FCX","COP","EOG","SLB","HAL",
                   "LMT","GD","NOC","CRWD","PANW","PLTR","NET","DDOG","RKLB","HII"]
    extras = ["SMCI","APP","CELH","DUOL","HUBS","WDAY","CYBR","IONQ","SOUN"]
    tickers = list(set(tickers + extras))
    end = datetime.now()
    start = end - timedelta(days=years*365)
    prices = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                         group_by="ticker", auto_adjust=True, threads=True)
    fundamentals = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            if not info or "marketCap" not in info: continue
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
        except: pass
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
def fetch_macro():
    """Fetch macro indicators from FRED. All timestamped."""
    api_key = os.environ.get("FRED_API_KEY")
    fallback = {"date":datetime.now().strftime("%Y-%m-%d"),
                "fed_funds_rate":4.5,"us_10y_yield":4.1,"yield_curve_slope":-0.4,
                "cpi_yoy":3.0,"oil_price":80,"vix":22,"credit_spread":1.5}
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
        return macro
    except: return fallback

# ── FX (exchangerate.host + yfinance fallback) ────────────────
def fetch_fx(base="EUR", targets=None):
    """Fetch FX rates. Tries exchangerate.host, falls back to yfinance. Timestamped."""
    if targets is None: targets = ["USD","GBP","CHF"]
    rates = {"date": datetime.now().strftime("%Y-%m-%d"), "base": base}
    # Try exchangerate.host (free, no key)
    try:
        url = f"https://api.exchangerate.host/latest?base={base}&symbols={','.join(targets)}"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        if data.get("success") or data.get("rates"):
            for t in targets:
                if t in data.get("rates",{}):
                    rates[f"{base}_{t}"] = round(data["rates"][t],4)
            return rates
    except: pass
    # Fallback: yfinance
    try:
        import yfinance as yf
        for t in targets:
            pair = f"{base}{t}=X"
            p = yf.Ticker(pair).info.get("regularMarketPrice")
            if p: rates[f"{base}_{t}"] = round(float(p),4)
    except: pass
    # Final fallback
    if f"{base}_USD" not in rates: rates[f"{base}_USD"] = 1.08
    if f"{base}_GBP" not in rates: rates[f"{base}_GBP"] = 0.86
    return rates

# ── COMBINED FETCH ────────────────────────────────────────────
def fetch_all_data(tickers=None, years=5, callback=None):
    """Fetch all data sources in one call. Returns dict with timestamps.
    Used by run_full_pipeline to get everything needed."""
    if callback: callback("Fetching universe...")
    universe_tickers, prices, fundamentals = fetch_universe(years)
    if tickers: universe_tickers = list(set(universe_tickers + tickers))

    if callback: callback("Fetching macro...")
    macro = fetch_macro()

    if callback: callback("Fetching FX...")
    fx = fetch_fx()

    if callback: callback("Fetching news sentiment...")
    sentiment = {}
    try:
        from .news import batch_sentiment
        sentiment = batch_sentiment(list(fundamentals.keys())[:150], callback)
    except: pass

    return {
        "tickers": universe_tickers,
        "prices": prices,
        "fundamentals": fundamentals,
        "macro": macro,
        "fx": fx,
        "sentiment": sentiment,
        "fetched_at": datetime.now().isoformat(),
    }
