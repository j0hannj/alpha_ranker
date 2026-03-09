"""
Alpha Ranker AI Agent v3
=========================
- Context-aware per tab
- Can execute actions: refresh prices, search news, run model, analyze, compare
- Understands the app architecture
- Works with Claude API or local Ollama
"""
import json, os, re
from pathlib import Path
from .news import build_news_context, search_general, get_sentiment_score

HISTORY_PATH = Path(__file__).parent.parent.parent / "db" / "agent_history.json"

def _load_history():
    if HISTORY_PATH.exists():
        try: return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except: pass
    return []

def _save_history(h):
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(h[-50:],ensure_ascii=False,indent=2),encoding="utf-8")

# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT — teaches the LLM about the app
# ═══════════════════════════════════════════════════════════════

APP_KNOWLEDGE = """You are Alpha Ranker AI, the built-in assistant of the Alpha Ranker desktop application.

== APP ARCHITECTURE ==
Alpha Ranker is a quantitative stock picking tool with these tabs:
1. PORTFOLIO: shows user's holdings (stocks + ETFs), P&L, sector exposure, historical value chart, and DCA projection
2. RANKINGS: ML model results ranking ~500 stocks by predicted 12M return. 4-model ensemble (LightGBM, XGBoost, Ridge, RandomForest)
3. INVEST: suggests how to allocate monthly DCA amount across ETFs and stock picks
4. AI AGENT: this chat interface
5. BACKTEST: walk-forward backtesting with OOS metrics (Rank IC, IR, L/S return, hit rate)
6. SETTINGS: API keys (Anthropic, FRED, FMP, Alpha Vantage) and preferences

== MODEL ARCHITECTURE (you must understand this deeply) ==

The alpha model is a cross-sectional equity factor model — at each point in time, it ranks
all ~500 stocks by predicted forward 12-month return. It is NOT a time series model.

ENSEMBLE: 4 models with different inductive biases, combined by IC-weighted average:
  1. LightGBM (gradient boosting): excels at non-linear interactions (e.g. energy×oil).
     Splits features greedily. Can overfit on noise but captures complex patterns.
  2. XGBoost (regularized boosting): similar to LightGBM but with different split strategy
     and regularization. Often complements LightGBM where it errs.
  3. Ridge (L2-penalized linear regression): captures monotone factor relationships only.
     Cannot model interactions. Acts as a sanity anchor — if trees diverge wildly from
     Ridge, the signal may be spurious.
  4. RandomForest (bagged trees): lower variance than boosting, more stable but less precise.
     Good at capturing robust patterns that don't depend on specific feature interactions.

FEATURE PIPELINE (this is how raw data becomes model input):
  Step 1: build_features_asof(date) — constructs features using ONLY data available at that date
    - Fundamental features: P/E, P/B, EV/EBITDA, ROE, margins, FCF yield, D/E, growth rates
      All computed as trailing-twelve-month (TTM) from the last 4 quarterly filings BEFORE the date
    - Technical features: momentum_12_1 (Jegadeesh-Titman), returns at 1/3/6/12M, volatility,
      Sharpe ratio, price vs MA50/MA200, drawdown from high, distance from low
    - Macro interactions: tech_x_rates (tech stocks penalized when rates rise),
      energy_x_oil (energy stocks boosted when oil rises), defensive_x_vix, financial_x_curve

  Step 2: rank_features() — cross-sectional percentile rank transform
    Each feature is converted to a 0-1 percentile rank within the cross-section.
    This removes outliers and makes features comparable (a P/E of 5 and a momentum of 0.3
    become comparable on the same 0-1 scale).

  Step 3: sector_neutralize() — subtract sector mean from each feature
    Ensures the model doesn't just bet on sectors. A tech stock with high momentum is only
    interesting if its momentum is high RELATIVE TO OTHER TECH STOCKS, not just vs. utilities.

  Step 4: predict_ensemble() — each model predicts, results combined by IC weight
    The model with the best out-of-sample Information Coefficient gets the most weight.

OUTPUT: alpha_score (raw prediction), alpha_rank (1 = best), confidence (z-score)

EXPLAINING INDIVIDUAL PREDICTIONS:
When asked "why is XOM ranked #3?", the system can compute per-feature contributions:
  - LightGBM/XGBoost: native tree SHAP values (pred_contrib=True) — exact decomposition
    of the prediction into per-feature contributions that sum to the final score
  - Ridge: coefficient × feature value — exact linear decomposition
  - RandomForest: feature_importance × deviation from median — approximate
These contributions tell you EXACTLY how much each feature pushed the score up or down.

MODEL CONSENSUS: when all 4 models rank a stock highly, it's a strong signal.
When they disagree (e.g. LightGBM says #2 but Ridge says #80), the signal is uncertain —
it likely depends on non-linear interactions that only the trees capture.

WALK-FORWARD BACKTESTING:
Train on [T-5y, T], predict [T, T+12M], slide by 1 quarter. Accumulate OOS predictions.
OOS metrics: Rank IC, IC IR, Long-Short quintile return, Hit Rate.
A Rank IC of 0.05+ is exploitable. IC IR of 0.5+ is good. Hit Rate of 55%+ is consistent.

When the user asks about any stock's ranking, ALWAYS explain:
1. What the model predicts and with what confidence
2. Which features are driving the prediction (using the contribution data if available)
3. Whether the 4 models agree or disagree
4. What the raw feature values mean in context (e.g. "P/E of 12 is cheap for tech")

When contribution data is provided in the context for a stock, USE IT naturally in your response.
You don't need to dump all the numbers — pick the 3-5 most impactful drivers and weave them
into your analysis like a quant analyst would in a research note. If the data is there but
the user didn't ask about that specific stock, don't force it — use it only when relevant.

== PROJECTION PIPELINE (EXPLAINABILITY) ==
Forward price projections are computed from the model prediction, NOT by guessing or clipping.

1. MODEL OUTPUT: The model predicts a single number per stock — the SIMPLE return over the
   training horizon (default 12 months). This is stored as predicted_return_pct (e.g. 12 = 12%
   over 12 months). It is NOT annualized and must NEVER be interpreted as a 2-year return.

2. RETURN TO PRICE: projected_price = current_price * (1 + r_H)^(display_months / H)
   where r_H = model predicted return over H months (decimal). For 12M: factor = 1+r. For 24M:
   factor = (1+r)^2 (same 12M return compounded). A 1-year prediction is never used as 2-year.

3. VARIABLE TRACE: For each holding the system can expose: model_horizon_months (H),
   model_return_pct, formula used, and per-horizon factor. Use this to answer "how was the
   projection computed?" or "why does X show -Y%?" — explain which variables were used and
   how they were transformed. If the model predicted an extreme return (e.g. -81%), the
   projection is faithfully applying it; the cause is the model output, not the projection.

4. MULTI-HORIZON: Today the model has one horizon (12M). Future versions may have
   predicted_return_3m, _6m, _12m, _24m so each display horizon uses its own prediction.

== AVAILABLE ACTIONS ==
You can request actions by including these tags in your response:
[ACTION:refresh_prices] - Refresh all portfolio prices from Yahoo Finance
[ACTION:search_news:TICKER] - Search recent news for a specific ticker
[ACTION:run_model] - Run the full ML pipeline (takes ~15 min)
[ACTION:analyze:TICKER] - Deep analysis of a specific stock
[ACTION:compare:TICKER1:TICKER2] - Compare two stocks
[ACTION:project:TICKER] - Monte Carlo price projection for a holding
[ACTION:fill_missing:TICKER:FIELD] - Search web to fill missing data for a ticker

== MISSING DATA PROTOCOL ==
If you detect that a ticker has missing fundamentals, ratios, or data (shown as None/NaN
in the context), you should:
1. Identify what data is missing
2. Use [ACTION:fill_missing:TICKER:FIELD] to search for it
3. Explain to the user what was missing and what you found
This makes you act as a quant research assistant that fills data gaps autonomously.

When you use an action, explain what you're doing and why.

== CRITICAL BEHAVIOR ==
You ALREADY HAVE the user's full portfolio data, holdings, P&L, sector exposure,
model predictions, and macro context loaded below. NEVER ask the user for data
you already have. If they ask "when do I reach 1M?", COMPUTE IT using their actual
portfolio value and DCA amount from the data below. Do NOT ask them to provide numbers
you already know. Be proactive: use the data, do the math, give the answer.

== RULES ==
- Be direct, concrete, give numbers FROM THE DATA YOU HAVE
- You are NOT a licensed financial advisor — quantitative analysis only
- When discussing model results, mention which models agree/disagree
- Always consider the user's current sector exposure
- Respond in the same language the user uses
- NEVER ask the user for information that is already in your context

== FORMATTING (STRICTLY ENFORCED) ==
You are displayed in a MONOSPACE PLAIN TEXT widget. ABSOLUTELY NO MARKDOWN.
VIOLATION EXAMPLES (NEVER DO THIS):
  BAD: **bold text**
  BAD: ### Header
  BAD: | col1 | col2 |
  BAD: ```code```
  BAD: *italic*
  BAD: - [x] checkbox

CORRECT FORMAT:
  CAPS for emphasis instead of bold
  --- SECTION NAME --- for headers
  Dashes for lists
  Aligned spaces for tables:
    Ticker   Return   Conv
    XOM      +18.2%   High
    LLY      +15.1%   Med+
  Keep responses under 250 words unless asked for detail.
"""

def build_context(portfolio_pnl=None, model_results=None, macro=None,
                  feat_imp=None, model_info=None, active_tab=None, extra=None):
    """Build system prompt with full context based on active tab."""
    ctx = APP_KNOWLEDGE + "\n"

    # Always include portfolio summary
    if portfolio_pnl:
        ctx += f"\n== USER PORTFOLIO (YOU HAVE THIS DATA — USE IT) ==\n"
        ctx += f"Total value: {portfolio_pnl['total_value']:,.0f} EUR\n"
        ctx += f"Total cost: {portfolio_pnl['total_cost']:,.0f} EUR\n"
        ctx += f"P&L: {portfolio_pnl['total_pnl']:+,.0f} EUR ({portfolio_pnl['total_pnl_pct']:+.1f}%)\n"
        for h in portfolio_pnl.get("holdings",[]):
            ctx += f"  {h['ticker']}: {h['units']}u, cost {h['avg_price']}, cur {h.get('current_price','?')}, P&L {h.get('pnl_pct',0):+.1f}%, val {h.get('value',0):,.0f}EUR\n"
        ctx += "Sectors: " + ", ".join(f"{s['sector']}:{s['pct']:.0f}%" for s in portfolio_pnl.get("sectors",[])[:6]) + "\n"

    # User settings & profile (all from Settings tab, nothing hardcoded)
    try:
        from . import portfolio as _pf
        ctx += f"\n== USER PROFILE & SETTINGS ==\n"
        settings_map = [
            ("monthly_dca", "Monthly DCA", "2200", "EUR"),
            ("base_currency", "Base Currency", "EUR", None),
            ("risk_profile", "Risk Profile", None, None),
            ("target_amount", "Wealth Target", None, "EUR"),
            ("investment_horizon", "Horizon", None, "years"),
            ("user_age", "Age", None, None),
            ("user_country", "Country", None, None),
            ("tax_notes", "Tax Rules", None, None),
            ("user_language", "Preferred Language", None, None),
            ("user_notes", "Custom Notes", None, None),
        ]
        for key, label, default, unit in settings_map:
            val = _pf.get_setting(key, default)
            if val:
                suffix = f" {unit}" if unit else ""
                ctx += f"  {label}: {val}{suffix}\n"
    except: pass

    # Tab-specific context
    if active_tab == "Portfolio":
        ctx += "\n== ACTIVE TAB: PORTFOLIO ==\n"
        ctx += "User is looking at their holdings, P&L, sector exposure, and value chart.\n"
        ctx += "Help with: position analysis, rebalancing, risk assessment, sector concentration.\n"

    elif active_tab == "Rankings" and model_results is not None:
        ctx += f"\n== ACTIVE TAB: RANKINGS (top 20) ==\n"
        for _,r in model_results.head(20).iterrows():
            line = f"  #{int(r['rank'])} {r['ticker']} ({r.get('sector','')}) +{r['predicted_return_pct']:.1f}%"
            line += f" conf={r['confidence']:.1f}σ"
            if r.get("recommendation") and r["recommendation"]==r["recommendation"]:
                line += f" analyst={r['recommendation']}"
            # Show per-model agreement
            model_preds = []
            for col in r.index:
                if col.startswith("pred_"):
                    model_preds.append(f"{col.replace('pred_','')[:3]}={r[col]:+.1f}%")
            if model_preds: line += f" [{', '.join(model_preds)}]"
            ctx += line + "\n"
        ctx += "Help with: why a stock is ranked high/low, model agreement, sector rotation.\n"

    elif active_tab == "Invest":
        ctx += "\n== ACTIVE TAB: INVEST ==\n"
        ctx += "User wants to allocate their monthly DCA. Consider current exposure + model picks.\n"

    elif active_tab == "Backtest" and model_info:
        ctx += f"\n== ACTIVE TAB: BACKTEST ==\n"
        for k,v in model_info.items():
            if k in ("per_model","ls_returns_series","ic_series"): continue
            ctx += f"  {k}: {v}\n"
        if "per_model" in model_info:
            ctx += "Per-model breakdown:\n"
            for name,m in model_info["per_model"].items():
                ctx += f"  {name}: IC={m.get('rank_ic',m.get('cv_r2','?'))}, weight={m.get('weight',0):.0%}\n"
        ctx += "Help with: interpreting metrics, model reliability, what to improve.\n"

    # Macro
    if macro:
        ctx += "\n== MACRO ==\n"
        for k,v in macro.items():
            if isinstance(v,(int,float)): ctx += f"  {k}: {v}\n"

    # Feature importances
    if feat_imp is not None and len(feat_imp)>0:
        ctx += "\n== TOP FEATURES ==\n"
        for feat,imp in feat_imp.head(8).items():
            ctx += f"  {feat}: {imp:.4f}\n"

    # Projection pipeline (for explainability)
    if model_info and isinstance(model_info, dict) and model_info.get("prediction_horizon_months") is not None:
        H = model_info["prediction_horizon_months"]
        ctx += f"\n== PROJECTION CONFIG ==\n"
        ctx += f"  Model prediction horizon: {H} months. predicted_return_pct is the {H}-month simple return (not annualized).\n"
        ctx += f"  Projections: projected_price = current_price * (1 + r)^{{m/{H}}}. Never interpret 1-year as 2-year.\n"

    # Extra context (news, analysis results)
    if extra: ctx += "\n" + extra

    return ctx

# ═══════════════════════════════════════════════════════════════
# ACTION PARSER
# ═══════════════════════════════════════════════════════════════

def parse_actions(response_text):
    """Extract action requests from LLM response."""
    actions = []
    patterns = [
        (r'\[ACTION:refresh_prices\]', "refresh_prices", []),
        (r'\[ACTION:run_model\]', "run_model", []),
        (r'\[ACTION:search_news:([A-Z0-9.-]+)\]', "search_news", None),
        (r'\[ACTION:analyze:([A-Z0-9.-]+)\]', "analyze", None),
        (r'\[ACTION:compare:([A-Z0-9.-]+):([A-Z0-9.-]+)\]', "compare", None),
        (r'\[ACTION:project:([A-Z0-9.-]+)\]', "project", None),
        (r'\[ACTION:fill_missing:([A-Z0-9.-]+):(\w+)\]', "fill_missing", None),
    ]
    for pattern, action_type, default_args in patterns:
        for match in re.finditer(pattern, response_text):
            args = list(match.groups()) if default_args is None else default_args
            actions.append({"type": action_type, "args": args})
    return actions

def clean_response(text):
    """Remove action tags from visible response."""
    return re.sub(r'\[ACTION:[^\]]+\]', '', text).strip()

# ═══════════════════════════════════════════════════════════════
# NEWS ENRICHMENT
# ═══════════════════════════════════════════════════════════════

def _extract_tickers(text):
    words = re.findall(r'\b([A-Z]{1,5}(?:-[A-Z])?)\b', text.upper())
    stop = {"LE","LA","LES","UN","UNE","DES","MON","TON","SON","ET","OU","QUI","QUE",
            "EST","SONT","DANS","POUR","PAR","SUR","AVEC","PAS","PLUS","TOP","EUR","USD",
            "DCA","ETF","AI","IA","BAS","MES","THE","AND","FOR","NOT","HAS","WAS","ARE"}
    return [w for w in words if w not in stop and len(w)>=2]

def _detect_search(text):
    triggers = ["oil","petrole","fed","taux","inflation","recession","guerre","iran",
                "chine","trump","tarif","crypto","bitcoin","marche","crash","news","sector"]
    return any(t in text.lower() for t in triggers)

def _enrich_with_news(user_message, portfolio_pnl=None):
    extra = ""
    tickers = _extract_tickers(user_message)
    if any(w in user_message.lower() for w in ["portfolio","portefeuille","positions","holdings"]):
        if portfolio_pnl:
            tickers += [h["ticker"] for h in portfolio_pnl.get("holdings",[])[:5]]
    seen = set()
    for t in tickers[:3]:
        if t in seen: continue; seen.add(t)
        try: extra += build_news_context(t)
        except: pass
    if _detect_search(user_message):
        try: extra += search_general(user_message[:80]+" stock market 2026")
        except: pass
    return extra

# ═══════════════════════════════════════════════════════════════
# LLM BACKENDS
# ═══════════════════════════════════════════════════════════════

def _chat_claude(msg, system, history, api_key):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    messages = history[-10:]+[{"role":"user","content":msg}]
    r = client.messages.create(model="claude-sonnet-4-20250514",max_tokens=2000,system=system,messages=messages)
    return r.content[0].text

def _chat_ollama(msg, system, history, model_name):
    import urllib.request
    messages = [{"role":"system","content":system}]
    for m in history[-10:]: messages.append(m)
    messages.append({"role":"user","content":msg})
    payload = json.dumps({"model":model_name,"messages":messages,"stream":False,
                          "options":{"temperature":0.7,"num_predict":2000}}).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/chat",data=payload,
                                headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode()).get("message",{}).get("content","No response.")

def _check_ollama():
    try:
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/tags"),timeout=2) as r:
            models=[m["name"] for m in json.loads(r.read().decode()).get("models",[])]
            for p in ["mistral-small","qwen2.5:14b","llama3.1","phi4","gemma3:12b","mistral","llama3"]:
                if p in models: return p
            return models[0] if models else None
    except: return None

# ═══════════════════════════════════════════════════════════════
# MAIN CHAT
# ═══════════════════════════════════════════════════════════════

def chat(user_message, portfolio_pnl=None, model_results=None, macro=None,
         feat_imp=None, model_info=None, api_key=None, active_tab=None,
         extra_context=None, model_state=None):
    """Main chat entry point. model_state = dict with models_dict, medians, feat_cols,
    prices, fundamentals_db, sector_map for generating per-stock explanations.
    Returns (response_text, actions_list)."""
    news_ctx = _enrich_with_news(user_message, portfolio_pnl)

    # Always enrich with model explanations for any mentioned tickers.
    # The LLM decides when and how to use this data — no keyword gating.
    explain_ctx = ""
    if model_state and model_state.get("models_dict"):
        tickers = _extract_tickers(user_message)
        # Also check if user refers to portfolio holdings by name or indirectly
        if portfolio_pnl:
            msg_lower = user_message.lower()
            for h in portfolio_pnl.get("holdings", []):
                tk = h["ticker"]
                name = h.get("name", "").lower()
                # Match ticker or company name fragments
                if tk.upper() in user_message.upper():
                    tickers.append(tk)
                elif name and any(w in msg_lower for w in name.split()[:2] if len(w) > 3):
                    tickers.append(tk)

        for t in list(dict.fromkeys(tickers))[:3]:  # Dedupe, max 3
            try:
                from .model import explain_ranking_context
                exp = explain_ranking_context(
                    t, model_state["models_dict"], model_state["medians"],
                    model_state["feat_cols"], model_state["prices"],
                    model_state["fundamentals_db"], macro or {},
                    model_state["sector_map"], model_state.get("yf_info"))
                explain_ctx += exp
            except: pass

    full_extra = news_ctx + explain_ctx + (extra_context or "")
    system = build_context(portfolio_pnl, model_results, macro, feat_imp,
                           model_info, active_tab, full_extra)
    history = _load_history()
    engine = None
    try:
        if api_key:
            response = _chat_claude(user_message, system, history, api_key)
            engine = "Claude Sonnet"
        else:
            m = _check_ollama()
            if m:
                response = _chat_ollama(user_message, system, history, m)
                engine = f"Ollama ({m})"
            else:
                return ("No AI engine available.\n\n"
                        "Option 1 — Claude: enter API key in Settings\n"
                        "Option 2 — Local: click 'Install' to set up Ollama", [])

        history.append({"role":"user","content":user_message})
        history.append({"role":"assistant","content":response})
        _save_history(history)

        actions = parse_actions(response)
        clean = clean_response(response)
        n_news = news_ctx.count("[") if news_ctx else 0
        header = f"[{engine}]"
        if n_news>0: header += f" + {max(n_news,1)} web"
        if explain_ctx: header += " + model explain"
        if actions: header += f" + {len(actions)} action(s)"
        return f"{header}\n\n{clean}", actions

    except Exception as e:
        return f"Error: {str(e)}", []

def get_engine_status(api_key=None):
    if api_key: return "claude","Claude Sonnet (API)"
    m = _check_ollama()
    if m: return "ollama",f"Ollama ({m})"
    return "none","No AI engine"

def get_history(): return _load_history()
def clear_history(): _save_history([])
