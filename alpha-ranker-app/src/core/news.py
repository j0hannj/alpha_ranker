"""News & web search module — 100% gratuit, zero API key.
Sources: yfinance news + DuckDuckGo search."""

import json, re
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# 1. YFINANCE NEWS
# ─────────────────────────────────────────────────────────────

def get_ticker_news(ticker, max_results=5):
    """Get recent news for a ticker via yfinance. Free, no key."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return []
        results = []
        for n in news[:max_results]:
            results.append({
                "title": n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "link": n.get("link", ""),
                "date": datetime.fromtimestamp(n.get("providerPublishTime", 0)).strftime("%Y-%m-%d") if n.get("providerPublishTime") else "",
                "type": n.get("type", ""),
                "source": "yfinance",
            })
        return results
    except:
        return []


# ─────────────────────────────────────────────────────────────
# 2. DUCKDUCKGO SEARCH
# ─────────────────────────────────────────────────────────────

def web_search(query, max_results=5):
    """Search the web via DuckDuckGo. Free, no API key."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", "")[:300],
                    "link": r.get("href", ""),
                    "source": "duckduckgo",
                })
            return results
    except ImportError:
        # Fallback: try without the package (won't work but handles gracefully)
        return []
    except:
        return []


def search_ticker(ticker, company_name=""):
    """Combined news + web search for a ticker."""
    results = {
        "ticker": ticker,
        "news": get_ticker_news(ticker),
        "web": web_search(f"{ticker} {company_name} stock analysis news 2026", max_results=4),
        "fetched_at": datetime.now().isoformat(),
    }
    return results


# ─────────────────────────────────────────────────────────────
# 3. SENTIMENT SCORING (simple keyword-based)
# ─────────────────────────────────────────────────────────────

POSITIVE = {"upgrade","buy","outperform","bullish","beat","surge","rally","strong",
            "growth","profit","record","exceeds","positive","optimistic","raises",
            "overweight","higher","boost","upside","momentum","accelerat","expand",
            "dividend","innovation","breakthrough","approve","launch","partner",
            "acquir","deal","revenue","earnings","impressive","exceed","top"}

NEGATIVE = {"downgrade","sell","underperform","bearish","miss","drop","crash","weak",
            "loss","decline","negative","pessimistic","cuts","lowers","underweight",
            "lower","risk","warning","debt","lawsuit","probe","investigate","layoff",
            "recession","default","bankrupt","delay","cancel","withdraw","fine",
            "penalty","fraud","concern","disappointing","slowdown","contraction"}

def compute_sentiment(texts):
    """Simple keyword sentiment score. Returns float in [-1, 1]."""
    if not texts:
        return 0.0
    pos_count = 0
    neg_count = 0
    total_words = 0
    for text in texts:
        words = set(re.findall(r'\w+', text.lower()))
        total_words += len(words)
        pos_count += len(words & POSITIVE)
        neg_count += len(words & NEGATIVE)
    if pos_count + neg_count == 0:
        return 0.0
    return round((pos_count - neg_count) / (pos_count + neg_count), 4)


def get_sentiment_score(ticker, company_name=""):
    """Get a sentiment score for a ticker from news + web."""
    data = search_ticker(ticker, company_name)
    texts = []
    for n in data.get("news", []):
        texts.append(n.get("title", ""))
    for w in data.get("web", []):
        texts.append(w.get("title", "") + " " + w.get("body", ""))
    score = compute_sentiment(texts)
    n_sources = len(data.get("news", [])) + len(data.get("web", []))
    return {"score": score, "n_sources": n_sources, "ticker": ticker}


# ─────────────────────────────────────────────────────────────
# 4. BATCH SENTIMENT (for model features)
# ─────────────────────────────────────────────────────────────

def batch_sentiment(tickers, callback=None):
    """Compute sentiment for a list of tickers. Returns dict {ticker: score}."""
    scores = {}
    for i, t in enumerate(tickers):
        if callback and (i + 1) % 25 == 0:
            callback(f"News sentiment: {i+1}/{len(tickers)}...")
        try:
            result = get_sentiment_score(t)
            scores[t] = result["score"]
        except:
            scores[t] = 0.0
    if callback:
        pos = sum(1 for s in scores.values() if s > 0)
        neg = sum(1 for s in scores.values() if s < 0)
        callback(f"Sentiment: {len(scores)} tickers ({pos} positifs, {neg} negatifs)")
    return scores


# ─────────────────────────────────────────────────────────────
# 5. CONTEXT BUILDER (for agent)
# ─────────────────────────────────────────────────────────────

def build_news_context(ticker, company_name=""):
    """Build a text summary of recent news for the agent."""
    data = search_ticker(ticker, company_name)
    ctx = f"\n== NEWS RECENTES: {ticker} ==\n"
    for n in data.get("news", [])[:4]:
        ctx += f"  [{n.get('date','')}] {n.get('title','')} — {n.get('publisher','')}\n"
    for w in data.get("web", [])[:3]:
        ctx += f"  [Web] {w.get('title','')}\n"
        if w.get("body"):
            ctx += f"        {w['body'][:200]}\n"
    sentiment = compute_sentiment(
        [n.get("title","") for n in data.get("news",[])] +
        [w.get("title","")+" "+w.get("body","") for w in data.get("web",[])]
    )
    ctx += f"\n  Sentiment score: {sentiment:+.2f} "
    if sentiment > 0.3: ctx += "(tres positif)"
    elif sentiment > 0: ctx += "(legerement positif)"
    elif sentiment < -0.3: ctx += "(tres negatif)"
    elif sentiment < 0: ctx += "(legerement negatif)"
    else: ctx += "(neutre)"
    ctx += "\n"
    return ctx


def search_general(query):
    """General web search for the agent (not ticker-specific)."""
    results = web_search(query, max_results=6)
    ctx = f"\n== RECHERCHE WEB: {query} ==\n"
    for r in results:
        ctx += f"  {r.get('title','')}\n"
        if r.get("body"):
            ctx += f"    {r['body'][:250]}\n"
    return ctx
