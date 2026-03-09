"""
Data Recovery Agent
====================
When data fetching fails (wrong ticker, missing price, API error),
this module uses the AI agent + web search to diagnose and fix the issue.

Architecture:
  error detected → build error context → search web → LLM analyzes → fix applied
"""
import json, re
from datetime import datetime

def resolve_ticker_error(ticker, error_msg, callback=None, isin=None):
    """When a ticker fails to fetch, try to fix it.
    If ISIN is provided, uses ISIN resolver first (most reliable).
    Then tries ticker variations, then web search."""
    if callback: callback(f"Recovering: {ticker} ({error_msg[:40]})")

    # 1. Try ISIN resolution first (most reliable)
    if isin:
        try:
            from .isin import resolve_isin
            result = resolve_isin(isin, callback)
            if result and result.get("ticker") != ticker:
                import yfinance as yf
                info = yf.Ticker(result["ticker"]).info
                p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
                if p and p > 0:
                    if callback: callback(f"  ISIN {isin} -> {result['ticker']}")
                    return {"original_ticker": ticker, "fixed_ticker": result["ticker"],
                            "price": round(float(p),2), "currency": info.get("currency","USD"),
                            "name": info.get("shortName",result["ticker"]),
                            "explanation": f"Resolved via ISIN {isin}",
                            "resolved_at": datetime.now().isoformat()}
        except: pass

    from .news import web_search

    # 2. Search for the correct ticker
    queries = [
        f"{ticker} stock yahoo finance ticker symbol",
        f"{ticker} ETF correct ticker exchange",
    ]
    results = []
    for q in queries:
        try:
            r = web_search(q, max_results=3)
            results.extend(r)
        except: pass

    if not results:
        if callback: callback(f"  No web results for {ticker}")
        return None

    # Try common ticker variations (limit to 5 to avoid blocking)
    variations = _generate_variations(ticker)[:5]

    # Try each variation with yfinance (fast fail on 404)
    import yfinance as yf
    for var in variations:
        try:
            tk = yf.Ticker(var)
            # Fast check: try fast_info first (doesn't make full API call)
            try:
                p = tk.fast_info.get("lastPrice") or tk.fast_info.get("previousClose")
            except:
                info = tk.info
                p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
                info_data = info
            else:
                info_data = tk.info if p else {}
            if p and p > 0:
                result = {
                    "original_ticker": ticker,
                    "fixed_ticker": var,
                    "price": round(float(p), 2),
                    "currency": info_data.get("currency", "USD"),
                    "name": info_data.get("shortName", var),
                    "explanation": f"'{ticker}' not found. Using '{var}' ({info_data.get('shortName','')}) instead.",
                    "resolved_at": datetime.now().isoformat(),
                }
                if callback: callback(f"  Fixed: {ticker} -> {var} ({result['name']}, {result['price']} {result['currency']})")
                return result
        except: continue

    # If variations didn't work, try to extract ticker from web results
    extracted = _extract_tickers_from_results(results, ticker)
    for ext in extracted:
        try:
            info = yf.Ticker(ext).info
            p = info.get("currentPrice") or info.get("regularMarketPrice")
            if p and p > 0:
                result = {
                    "original_ticker": ticker,
                    "fixed_ticker": ext,
                    "price": round(float(p), 2),
                    "currency": info.get("currency", "USD"),
                    "name": info.get("shortName", ext),
                    "explanation": f"Web search suggests '{ext}' for '{ticker}'.",
                    "resolved_at": datetime.now().isoformat(),
                }
                if callback: callback(f"  Web fix: {ticker} → {ext}")
                return result
        except: continue

    if callback: callback(f"  Could not resolve {ticker}")
    return None


def resolve_missing_price(ticker, currency="EUR", callback=None):
    """When yfinance fails for a ticker, try alternative sources via web search."""
    from .news import web_search

    if callback: callback(f"Searching price for {ticker}...")

    # Search for current price
    results = web_search(f"{ticker} current stock price today", max_results=5)

    # Try to extract price from search results
    for r in results:
        text = r.get("body", "") + " " + r.get("title", "")
        price = _extract_price_from_text(text)
        if price and price > 0:
            if callback: callback(f"  Found price from web: {ticker} = {price}")
            return {"ticker": ticker, "price": price, "currency": currency,
                    "source": "web_search", "resolved_at": datetime.now().isoformat()}

    return None


def resolve_missing_fundamental(ticker, field, callback=None):
    """When a fundamental data point is missing, search the web for it."""
    from .news import web_search

    field_queries = {
        "revenue": f"{ticker} annual revenue latest",
        "eps": f"{ticker} earnings per share EPS latest",
        "pe_ratio": f"{ticker} P/E ratio current",
        "market_cap": f"{ticker} market capitalization",
        "dividend_yield": f"{ticker} dividend yield current",
        "roe": f"{ticker} return on equity ROE",
        "debt_to_equity": f"{ticker} debt to equity ratio",
        "gross_margin": f"{ticker} gross margin percentage",
    }

    query = field_queries.get(field, f"{ticker} {field} financial data")
    results = web_search(query, max_results=5)

    for r in results:
        text = r.get("body", "") + " " + r.get("title", "")
        value = _extract_number_from_text(text, field)
        if value is not None:
            if callback: callback(f"  Found {field} for {ticker}: {value}")
            return {"ticker": ticker, "field": field, "value": value,
                    "source": "web_search", "resolved_at": datetime.now().isoformat()}

    return None


def batch_recover_prices(failed_tickers, callback=None, isin_map=None):
    """Try to recover prices for failed tickers.
    isin_map: dict {ticker: isin} for ISIN-based resolution.
    Returns dict {ticker: recovery_result}."""
    if isin_map is None: isin_map = {}
    recovered = {}
    for t in failed_tickers:
        isin = isin_map.get(t)
        fix = resolve_ticker_error(t, "price fetch failed", callback, isin=isin)
        if fix:
            recovered[t] = fix
            continue
        price = resolve_missing_price(t, callback=callback)
        if price:
            recovered[t] = price
    return recovered


# ── HELPERS ───────────────────────────────────────────────────

def _generate_variations(ticker):
    """Generate common ticker variations to try."""
    variations = []
    base = ticker.split(".")[0]

    # Exchange suffixes to try
    suffixes = ["", ".DE", ".AS", ".L", ".PA", ".MI", ".SW", ".TO", ".AX"]

    for s in suffixes:
        var = base + s
        if var != ticker:
            variations.append(var)

    # Common transforms
    if "-" in ticker:
        variations.append(ticker.replace("-", "."))
        variations.append(ticker.replace("-", ""))
    if "." in ticker:
        variations.append(ticker.replace(".", "-"))

    return variations


def _extract_tickers_from_results(results, original):
    """Try to extract likely ticker symbols from web search results."""
    tickers = set()
    base = original.split(".")[0]

    for r in results:
        text = r.get("body", "") + " " + r.get("title", "")
        # Look for patterns like (XXXX) or XXXX:NYSE etc
        patterns = re.findall(r'\(([A-Z]{1,6}(?:\.[A-Z]{1,3})?)\)', text)
        patterns += re.findall(r'([A-Z]{1,6}(?:\.[A-Z]{1,3})?):(?:NYSE|NASDAQ|AMEX|LSE|FRA|AMS)', text)
        for p in patterns:
            if p.startswith(base) or base in p:
                tickers.add(p)

    return list(tickers)[:5]


def _extract_price_from_text(text):
    """Extract a stock price from a text snippet."""
    # Look for patterns like $123.45, €123.45, 123.45 USD, etc
    patterns = [
        r'[\$€£](\d{1,6}[.,]\d{2})',           # $123.45
        r'(\d{1,6}[.,]\d{2})\s*(?:USD|EUR|GBP)', # 123.45 USD
        r'price[:\s]+(\d{1,6}[.,]\d{2})',        # price: 123.45
        r'trading at\s+(\d{1,6}[.,]\d{2})',      # trading at 123.45
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except: pass
    return None


def _extract_number_from_text(text, field):
    """Extract a numerical value for a specific field from text."""
    # General number extraction near field name
    field_words = field.replace("_", " ")
    idx = text.lower().find(field_words)
    if idx == -1:
        # Try variations
        for alt in [field.replace("_", ""), field]:
            idx = text.lower().find(alt.lower())
            if idx >= 0: break

    if idx >= 0:
        # Look for numbers near the field mention
        nearby = text[max(0, idx-20):idx+100]
        numbers = re.findall(r'[-]?\d+[.,]?\d*[%BMK]?', nearby)
        for n in numbers:
            try:
                clean = n.replace(",", ".").replace("%", "").replace("B", "e9").replace("M", "e6").replace("K", "e3")
                return float(clean)
            except: pass
    return None
