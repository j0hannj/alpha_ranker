"""
ISIN Resolver — Maps ISIN codes to Yahoo Finance tickers.

Strategy:
  1. Local cache (instant)
  2. OpenFIGI API (free, no key for <20 req/min)
  3. Web search fallback (DuckDuckGo)
  4. Brute-force exchange suffixes

ISINs are universal. Tickers are exchange-specific.
This module bridges the gap.
"""
import json, re, urllib.request
from pathlib import Path
from datetime import datetime

CACHE_FILE = Path(__file__).parent.parent.parent / "db" / "isin_cache.json"

# Known mappings (hardcoded for speed)
KNOWN = {
    "LU1681038243": "6AQQ.DE",    # Amundi Nasdaq-100 Swap ETF EUR
    "IE00B4L5Y983": "IWDA.AS",    # iShares Core MSCI World
    "IE00BK5BQT80": "VWCE.DE",    # Vanguard FTSE All-World
    "US0378331005": "AAPL",       # Apple
    "US5949181045": "MSFT",       # Microsoft
    "US67066G1040": "NVDA",       # NVIDIA
    "US02079K3059": "GOOGL",      # Alphabet A
    "US02079K1079": "GOOG",       # Alphabet C
    "US0231351067": "AMZN",       # Amazon
    "US0846707026": "BRK-B",      # Berkshire B
    "US11135F1012": "AVGO",       # Broadcom
    "US46266C1053": "IONQ",       # IonQ
    "US69608A1088": "PLTR",       # Palantir
    "US30303M1027": "META",       # Meta
    "US88160R1014": "TSLA",       # Tesla
    "US46625H1005": "JPM",        # JP Morgan
    "US91324P1021": "UNH",        # UnitedHealth
    "US30231G1022": "XOM",        # ExxonMobil
    "LU1829221024": "UST.PA",     # Amundi Core Nasdaq-100
    "LU1681038326": "10A4.DE",    # Amundi Nasdaq-100 USD
}


def _load_cache():
    if CACHE_FILE.exists():
        try: return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except: pass
    return {}

def _save_cache(cache):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def resolve_isin(isin, callback=None):
    """Resolve an ISIN to a Yahoo Finance ticker.
    Returns {"ticker": str, "name": str, "exchange": str, "source": str} or None."""
    isin = isin.strip().upper()

    # 1. Known hardcoded
    if isin in KNOWN:
        ticker = KNOWN[isin]
        if callback: callback(f"ISIN {isin} -> {ticker} (known)")
        return {"ticker": ticker, "source": "known", "isin": isin}

    # 2. Local cache
    cache = _load_cache()
    if isin in cache:
        if callback: callback(f"ISIN {isin} -> {cache[isin]['ticker']} (cached)")
        return cache[isin]

    # 3. OpenFIGI API (free, no key needed for low volume)
    result = _try_openfigi(isin, callback)
    if result:
        cache[isin] = result
        _save_cache(cache)
        return result

    # 4. Web search
    result = _try_web_search(isin, callback)
    if result:
        cache[isin] = result
        _save_cache(cache)
        return result

    # 5. Brute force: try ISIN-derived base + exchange suffixes
    result = _try_brute_force(isin, callback)
    if result:
        cache[isin] = result
        _save_cache(cache)
        return result

    if callback: callback(f"Could not resolve ISIN {isin}")
    return None


def _try_openfigi(isin, callback=None):
    """Query OpenFIGI API to map ISIN to ticker."""
    try:
        payload = json.dumps([{"idType": "ID_ISIN", "idValue": isin}]).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openfigi.com/v3/mapping",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        if data and data[0].get("data"):
            entries = data[0]["data"]
            # Prefer EUR exchanges, then USD
            preferred_exchanges = ["GY", "GR", "NA", "FP", "IM", "LN", "SM", "US", "UW", "UN", "UQ"]
            best = None
            for entry in entries:
                ticker = entry.get("ticker", "")
                exchange = entry.get("exchCode", "")
                name = entry.get("name", "")
                if not best:
                    best = (ticker, exchange, name)
                for pref in preferred_exchanges:
                    if exchange == pref:
                        best = (ticker, exchange, name)
                        break

            if best:
                ticker, exchange, name = best
                # Convert OpenFIGI exchange code to yfinance suffix
                yf_ticker = _figi_to_yfinance(ticker, exchange)
                if yf_ticker and _verify_yfinance(yf_ticker):
                    if callback: callback(f"ISIN {isin} -> {yf_ticker} via OpenFIGI ({name})")
                    return {"ticker": yf_ticker, "name": name, "exchange": exchange,
                            "source": "openfigi", "isin": isin}
    except Exception as e:
        if callback: callback(f"OpenFIGI error: {e}")
    return None


def _figi_to_yfinance(ticker, exchange_code):
    """Convert OpenFIGI exchange code to Yahoo Finance ticker suffix."""
    suffix_map = {
        "GY": ".DE", "GR": ".DE",  # Germany (Frankfurt/XETRA)
        "NA": ".AS",                 # Netherlands (Amsterdam)
        "FP": ".PA",                 # France (Paris)
        "IM": ".MI",                 # Italy (Milan)
        "LN": ".L",                  # London
        "SM": ".MC",                 # Spain (Madrid)
        "SW": ".SW",                 # Switzerland
        "US": "", "UW": "", "UN": "", "UQ": "",  # US exchanges
        "AU": ".AX",                 # Australia
        "CT": ".TO",                 # Canada (Toronto)
    }
    suffix = suffix_map.get(exchange_code, "")
    return f"{ticker}{suffix}" if ticker else None


def _verify_yfinance(ticker):
    """Quick verify that yfinance can find this ticker."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        return p is not None and p > 0
    except:
        return False


def _try_web_search(isin, callback=None):
    """Search web for ISIN to ticker mapping."""
    try:
        from .news import web_search
        results = web_search(f"{isin} yahoo finance ticker", max_results=5)
        # Extract potential tickers from results
        for r in results:
            text = r.get("title", "") + " " + r.get("body", "")
            # Look for patterns like (XXXX.XX) or XXXX.XX
            tickers = re.findall(r'\b([A-Z0-9]{1,6}\.[A-Z]{1,2})\b', text)
            tickers += re.findall(r'\(([A-Z0-9]{1,6}(?:\.[A-Z]{1,2})?)\)', text)
            for t in tickers:
                if _verify_yfinance(t):
                    if callback: callback(f"ISIN {isin} -> {t} via web search")
                    return {"ticker": t, "source": "web_search", "isin": isin}
    except:
        pass
    return None


def _try_brute_force(isin, callback=None):
    """Try to derive ticker from ISIN country code + common suffixes."""
    country = isin[:2]
    suffixes_by_country = {
        "US": [""],
        "IE": [".AS", ".DE", ".L"],
        "LU": [".DE", ".PA", ".AS", ".MI"],
        "GB": [".L"],
        "DE": [".DE"],
        "FR": [".PA"],
        "NL": [".AS"],
    }
    suffixes = suffixes_by_country.get(country, [".DE", ".AS", ".L", ".PA", ""])
    # Can't derive ticker from ISIN digits, skip brute force
    return None


def batch_resolve(isin_list, callback=None):
    """Resolve multiple ISINs. Returns dict {isin: result}."""
    results = {}
    for isin in isin_list:
        r = resolve_isin(isin, callback)
        if r:
            results[isin] = r
    return results
