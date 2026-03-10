"""
ISIN Resolver — Maps ISIN codes to Yahoo Finance tickers.

Strategy: KNOWN → cache → Yahoo Search (1 call, no API key) → FMP fallback.
Yahoo Finance search accepts ISINs directly. No OpenFIGI, no heavy verification.
"""
import json
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
CACHE_FILE = Path(__file__).parent.parent.parent / "db" / "isin_cache.json"

# European ETFs that can be problematic on Yahoo Search — keep explicit mapping
KNOWN = {
    "LU1681038243": "6AQQ.DE",
    "IE00B4L5Y983": "IWDA.AS",
    "IE00BK5BQT80": "VWCE.DE",
    "LU1829221024": "UST.PA",
    "LU1681038326": "10A4.DE",
}


def _load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("isin _load_cache: %s", e)
    return {}


def _save_cache(cache):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def resolve_isin(isin, callback=None):
    """Resolve ISIN to Yahoo Finance ticker.
    Returns {"ticker": str, "name": str, "source": str, "isin": str} or None."""
    isin = isin.strip().upper()

    # 1. Hardcoded (European ETFs)
    if isin in KNOWN:
        if callback:
            callback(f"{isin} -> {KNOWN[isin]} (known)")
        return {"ticker": KNOWN[isin], "source": "known", "isin": isin}

    # 2. Local cache
    cache = _load_cache()
    if isin in cache:
        if callback:
            callback(f"{isin} -> {cache[isin]['ticker']} (cached)")
        return cache[isin]

    # 3. Yahoo Finance search — primary resolver (1 call, no key)
    result = _yahoo_search(isin, callback)
    if result:
        cache[isin] = result
        _save_cache(cache)
        return result

    # 4. FMP fallback (if key configured)
    result = _fmp_search(isin, callback)
    if result:
        cache[isin] = result
        _save_cache(cache)
        return result

    if callback:
        callback(f"Could not resolve {isin}. Enter the Yahoo Finance ticker directly (e.g. AAPL, VWCE.DE).")
    return None


def _yahoo_search(isin, callback=None):
    """Yahoo Finance search — accepts ISINs, no API key."""
    try:
        # Method 1: yfinance Search (if available)
        try:
            import yfinance as yf
            s = yf.Search(isin)
            if hasattr(s, "quotes") and s.quotes:
                q = s.quotes[0] if isinstance(s.quotes, list) else list(s.quotes)[0]
                ticker = q.get("symbol") if isinstance(q, dict) else getattr(q, "symbol", None)
                name = q.get("shortname", "") if isinstance(q, dict) else getattr(q, "shortname", "")
                if ticker:
                    if callback:
                        callback(f"{isin} -> {ticker} via Yahoo Search")
                    return {"ticker": ticker, "name": name or "", "source": "yahoo_search", "isin": isin}
        except (AttributeError, TypeError, IndexError):
            pass

        # Method 2: raw HTTP to Yahoo search API
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={isin}&quotesCount=5&newsCount=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        quotes = data.get("quotes", [])
        if quotes:
            ticker = quotes[0].get("symbol")
            name = quotes[0].get("shortname", "") or quotes[0].get("longname", "")
            if ticker:
                if callback:
                    callback(f"{isin} -> {ticker} via Yahoo Search API")
                return {"ticker": ticker, "name": name, "source": "yahoo_search", "isin": isin}
    except Exception as e:
        if callback:
            callback(f"Yahoo search error: {e}")
    return None


def _fmp_search(isin, callback=None):
    """FMP search fallback — requires FMP_API_KEY."""
    import os
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        return None
    try:
        url = f"https://financialmodelingprep.com/api/v3/search?query={isin}&apikey={api_key}&limit=5"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
        if data:
            for item in data:
                ticker = item.get("symbol")
                if ticker:
                    if callback:
                        callback(f"{isin} -> {ticker} via FMP")
                    return {
                        "ticker": ticker,
                        "name": item.get("name", ""),
                        "source": "fmp",
                        "isin": isin,
                    }
    except Exception as e:
        if callback:
            callback(f"FMP search error: {e}")
    return None


def batch_resolve(isin_list, callback=None):
    """Resolve multiple ISINs. Returns dict {isin: result}."""
    return {isin: r for isin in isin_list if (r := resolve_isin(isin, callback))}
