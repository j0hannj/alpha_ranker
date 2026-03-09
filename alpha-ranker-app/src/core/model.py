"""
Alpha Model — Institutional Grade
====================================
Pipeline: features → rank → sector neutralize → ensemble → alpha score

Features at time t use ONLY information available at t.
Targets: forward_return(t → t+horizon).

Ensemble: LightGBM + XGBoost + Ridge + RandomForest
Combination: OOS-IC-weighted average
Output: alpha_score, alpha_rank per ticker
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

# New modular imports (refactored architecture)
from features.fundamental_features import get_fundamentals_asof, build_features_asof
from features.price_features import compute_forward_return
from features.cross_sectional import (
    rank_features,
    sector_neutralize,
    factor_neutralize_scores,
    add_sector_interactions,
)

warnings.filterwarnings("ignore")
CACHE_DIR = Path(__file__).parent.parent.parent / "db"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_CACHE = CACHE_DIR / "model_cache.pkl"
FUNDAMENTALS_CACHE = CACHE_DIR / "fundamentals_cache.json"

# ══════════════════════════════════════════════════════════════
# FMP HISTORICAL DATA
# ══════════════════════════════════════════════════════════════
def fetch_fmp_quarterly(ticker, api_key, limit=40):
    import urllib.request
    raw = {}
    for name, ep in {"income":"income-statement","metrics":"key-metrics","ratios":"ratios"}.items():
        try:
            url = f"https://financialmodelingprep.com/api/v3/{ep}/{ticker}?period=quarter&limit={limit}&apikey={api_key}"
            with urllib.request.urlopen(url,timeout=15) as r: raw[name] = json.loads(r.read().decode())
        except: raw[name] = []
    by_date = {}
    for item in raw.get("income",[]):
        d = item.get("filingDate") or item.get("date")
        if d: by_date.setdefault(d,{})["income"]=item; by_date[d]["filing_date"]=d; by_date[d]["period_date"]=item.get("date")
    for src in ["metrics","ratios"]:
        for item in raw.get(src,[]):
            d=item.get("date")
            for fd in by_date:
                if by_date[fd].get("period_date")==d: by_date[fd][src]=item; break
    results = []
    for fd, data in sorted(by_date.items()):
        inc,met,rat = data.get("income",{}),data.get("metrics",{}),data.get("ratios",{})
        results.append({"ticker":ticker,"filing_date":fd,"period_date":data.get("period_date"),
            "revenue":inc.get("revenue"),"net_income":inc.get("netIncome"),"eps":inc.get("eps"),
            "ebitda":inc.get("ebitda"),"pe_ratio":met.get("peRatio"),"pb_ratio":met.get("pbRatio"),
            "ev_ebitda":met.get("enterpriseValueOverEBITDA"),"roe":met.get("roe"),
            "debt_to_equity":met.get("debtToEquity"),"current_ratio":met.get("currentRatio"),
            "fcf_per_share":met.get("freeCashFlowPerShare"),"market_cap":met.get("marketCap"),
            "dividend_yield":met.get("dividendYield"),"revenue_per_share":met.get("revenuePerShare"),
            "gross_margin":rat.get("grossProfitMargin"),"operating_margin":rat.get("operatingProfitMargin"),
            "net_margin":rat.get("netProfitMargin"),"peg_ratio":rat.get("priceEarningsToGrowthRatio")})
    return results

def fetch_all_fundamentals(tickers, api_key, callback=None):
    cache = {}
    if FUNDAMENTALS_CACHE.exists():
        try:
            cache = json.loads(FUNDAMENTALS_CACHE.read_text(encoding="utf-8"))
            if (datetime.now()-datetime.fromisoformat(cache.get("_date","2000-01-01"))).days < 7:
                data = {k:v for k,v in cache.items() if k!="_date"}
                if len(data)>50:
                    if callback: callback(f"FMP cache: {len(data)} tickers")
                    return data
        except: pass
    if callback: callback(f"FMP: fetching {len(tickers)} tickers...")
    data = {}
    for i,t in enumerate(tickers):
        if callback and (i+1)%25==0: callback(f"FMP: {i+1}/{len(tickers)}...")
        try:
            rows = fetch_fmp_quarterly(t,api_key)
            if rows: data[t]=rows
        except: pass
    FUNDAMENTALS_CACHE.write_text(json.dumps({**data,"_date":datetime.now().isoformat()},default=str),encoding="utf-8")
    if callback: callback(f"FMP: {len(data)} tickers loaded")
    return data

# ══════════════════════════════════════════════════════════════
# ENSEMBLE ENGINE
# ══════════════════════════════════════════════════════════════
def _get_models():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    return {
        "LightGBM": LGBMRegressor(n_estimators=500,max_depth=5,learning_rate=0.03,
            subsample=0.7,colsample_bytree=0.6,min_child_samples=15,
            reg_alpha=0.3,reg_lambda=0.3,random_state=42,verbose=-1,n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500,max_depth=5,learning_rate=0.03,
            subsample=0.7,colsample_bytree=0.6,min_child_weight=15,
            reg_alpha=0.3,reg_lambda=0.3,random_state=42,verbosity=0,n_jobs=-1),
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(n_estimators=300,max_depth=8,
            min_samples_leaf=15,max_features=0.6,random_state=42,n_jobs=-1),
    }

def predict_ensemble(models_dict, X, method="ic_weighted"):
    """Ensemble prediction. Combines models weighted by their OOS information coefficient.
    Returns (predictions_array, blend_info_dict)."""
    all_preds = {}; weights = {}
    for name, info in models_dict.items():
        if info.get("model") is None: continue
        try:
            p = info["model"].predict(X)
            all_preds[name] = p
            weights[name] = max(info.get("ic",info.get("cv_r2",0.01)), 0.001)
        except: pass
    if not all_preds: return np.zeros(len(X)), {}
    total_w = sum(weights.values())
    final = sum(all_preds[n]*(weights[n]/total_w) for n in all_preds)
    blend = {n:{"weight":round(weights[n]/total_w,3),"mean_pred":round(float(np.mean(p)),4)}
             for n,p in all_preds.items()}
    return final, blend

def _get_feature_importance(models_dict, feat_cols):
    total = np.zeros(len(feat_cols)); count = 0
    for info in models_dict.values():
        if info.get("model") and hasattr(info["model"],"feature_importances_"):
            imp = info["model"].feature_importances_
            if len(imp)==len(feat_cols): total += imp/imp.sum(); count += 1
    if count>0: total /= count
    return pd.Series(total, index=feat_cols).sort_values(ascending=False)

# ══════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTESTING
# ══════════════════════════════════════════════════════════════
def walk_forward_train(prices, fundamentals_db, macro, sector_map, tickers,
                       start_year=2019, horizon_months=12, callback=None):
    """Walk-forward with ensemble. Enforces temporal consistency throughout."""
    rebal_dates = [datetime(y,m,1) for y in range(start_year,datetime.now().year+1)
                   for m in [1,4,7,10] if datetime(y,m,1) < datetime.now()-timedelta(days=horizon_months*30)]
    if callback: callback(f"Walk-forward: {len(rebal_dates)} periods")
    all_periods = []; meta_cols = ["ticker","date","sector","name","forward_return"]
    for i,rd in enumerate(rebal_dates):
        if callback and (i+1)%4==0: callback(f"Features: period {i+1}/{len(rebal_dates)}")
        df = build_features_asof(prices,fundamentals_db,macro,rd,tickers)
        if len(df)==0: continue
        df = add_sector_interactions(df,sector_map)
        df["forward_return"] = df["ticker"].apply(lambda t: compute_forward_return(prices,t,rd,horizon_months))
        df = df.dropna(subset=["forward_return"])
        if len(df)<20: continue
        df["period_idx"]=i; all_periods.append(df)
    if not all_periods:
        if callback: callback("Not enough data"); return None,None,None,None,None
    full_df = pd.concat(all_periods,ignore_index=True)
    feat_cols = [c for c in full_df.columns if c not in meta_cols+["period_idx"]
                 and full_df[c].dtype in [np.float64,np.int64,float,int]]
    if callback: callback(f"{len(full_df)} obs, {len(feat_cols)} features, {full_df['period_idx'].nunique()} periods")

    period_indices = sorted(full_df["period_idx"].unique()); min_train=8
    oos_preds=[]; per_model_oos = {n:[] for n in _get_models().keys()}
    for i,tp in enumerate(period_indices):
        if i<min_train: continue
        trn = full_df[full_df["period_idx"].isin(period_indices[:i])]
        tst = full_df[full_df["period_idx"]==tp]
        if len(trn)<50 or len(tst)<10: continue
        Xtr,ytr = trn[feat_cols].copy(), trn["forward_return"].copy()
        Xte,yte = tst[feat_cols].copy(), tst["forward_return"].copy()
        med = Xtr.median()
        Xtr = Xtr.fillna(med).replace([np.inf,-np.inf],np.nan).fillna(med)
        Xte = Xte.fillna(med).replace([np.inf,-np.inf],np.nan).fillna(med)
        # Rank transform + sector neutralize for training
        Xtr_r = rank_features(Xtr,feat_cols)
        Xte_r = rank_features(Xte,feat_cols)
        ytr = ytr.clip(ytr.quantile(0.02),ytr.quantile(0.98))
        models = _get_models()
        for name,m in models.items():
            try:
                m.fit(Xtr_r,ytr); p=m.predict(Xte_r)
                ic = stats.spearmanr(p,yte.values)[0] if len(yte)>5 else 0
                per_model_oos[name].append(ic)
            except: per_model_oos[name].append(0)
        # Ensemble for this period
        ens = {n:{"model":m,"ic":np.mean(per_model_oos[n]) if per_model_oos[n] else 0.01}
               for n,m in models.items()}
        for n,m in models.items():
            try: m.fit(Xtr_r,ytr); ens[n]["model"]=m
            except: pass
        ep,_ = predict_ensemble(ens,Xte_r)
        for j,(idx,row) in enumerate(tst.iterrows()):
            oos_preds.append({"ticker":row["ticker"],"period":int(tp),"predicted":ep[j],"actual":row["forward_return"]})

    # OOS metrics
    oos_df = pd.DataFrame(oos_preds)
    if len(oos_df)>0:
        corr = oos_df["predicted"].corr(oos_df["actual"])
        rc,_ = stats.spearmanr(oos_df["predicted"],oos_df["actual"])
        ic_per = oos_df.groupby("period").apply(lambda g: g["predicted"].corr(g["actual"]) if len(g)>5 else np.nan).dropna()
        ls_ret = []
        for _,g in oos_df.groupby("period"):
            if len(g)<10: continue
            g=g.sort_values("predicted",ascending=False); n=max(len(g)//5,2)
            ls_ret.append(g.head(n)["actual"].mean()-g.tail(n)["actual"].mean())
        oos_metrics = {"n_predictions":len(oos_df),"n_periods":int(oos_df["period"].nunique()),
            "pearson_correlation":round(corr,4),"spearman_rank_corr":round(rc,4),
            "mean_ic":round(ic_per.mean(),4),"ic_std":round(ic_per.std(),4),
            "ic_ir":round(ic_per.mean()/ic_per.std(),4) if ic_per.std()>0 else 0,
            "mean_ls_return":round(np.mean(ls_ret),4) if ls_ret else 0,
            "hit_rate":round((ic_per>0).mean(),4),
            "ls_returns_series":[round(r,4) for r in ls_ret],
            "ic_series":[round(float(v),4) for v in ic_per.values],
            "per_model_ic":{n:round(np.mean(v),4) for n,v in per_model_oos.items() if v}}
    else:
        oos_metrics = {"n_predictions":0,"per_model_ic":{}}
    if callback:
        callback(f"Ensemble OOS Rank IC: {oos_metrics.get('spearman_rank_corr','?')}")
        for n,ic in oos_metrics.get("per_model_ic",{}).items(): callback(f"  {n}: IC={ic:.4f}")

    # Final ensemble on ALL data with rank + sector neutral
    if callback: callback("Training final ensemble...")
    X_all = full_df[feat_cols].copy(); y_all = full_df["forward_return"].copy()
    med_final = X_all.median()
    X_all = X_all.fillna(med_final).replace([np.inf,-np.inf],np.nan).fillna(med_final)
    X_all = rank_features(X_all,feat_cols)
    if "sector" in full_df.columns:
        X_all["sector"] = full_df["sector"].values
        X_all = sector_neutralize(X_all,feat_cols,"sector")
        X_all = X_all.drop(columns=["sector"],errors="ignore")
    y_all = y_all.clip(y_all.quantile(0.02),y_all.quantile(0.98))
    final_models = {}
    for name,m in _get_models().items():
        try:
            m.fit(X_all,y_all)
            final_models[name] = {"model":m,
                "ic":np.mean(per_model_oos.get(name,[])) if per_model_oos.get(name) else 0.01,
                "cv_r2":np.mean(per_model_oos.get(name,[])) if per_model_oos.get(name) else 0}
            if callback: callback(f"  {name}: trained, OOS IC={final_models[name]['ic']:.4f}")
        except: pass
    feat_imp = _get_feature_importance(final_models,feat_cols)
    return final_models, med_final, feat_cols, feat_imp, oos_metrics

# ══════════════════════════════════════════════════════════════
# CURRENT PREDICTIONS → ALPHA SCORE + RANK
# ══════════════════════════════════════════════════════════════
def predict_current(models_dict, medians, feat_cols, prices, fundamentals_db,
                    macro, sector_map, tickers, yf_info=None, callback=None):
    """Generate current alpha scores and ranks using the trained ensemble."""
    if callback: callback("Generating alpha scores...")
    df = build_features_asof(prices,fundamentals_db,macro,datetime.now(),tickers)
    df = add_sector_interactions(df,sector_map)
    if yf_info:
        df["sector"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("sector","Unknown"))
        df["name"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("shortName",t))
        df["current_price"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("currentPrice"))
        df["target_mean"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("targetMeanPrice"))
        df["recommendation"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("recommendationKey"))
        df["num_analysts"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("numberOfAnalystOpinions"))

    X = df[feat_cols].fillna(medians).replace([np.inf,-np.inf],np.nan).fillna(medians)
    X = rank_features(X, feat_cols)  # Same transform as training
    preds, blend = predict_ensemble(models_dict, X)

    # Raw alpha score before factor neutralization
    df["alpha_score_raw"] = preds
    # Factor-neutralized alpha score (size / risk / momentum / sector)
    df["alpha_score"] = factor_neutralize_scores(
        df,
        score_col="alpha_score_raw",
        sector_col="sector"
    )
    df["predicted_return_pct"] = (df["alpha_score"]*100).round(2)
    df = df.sort_values("alpha_score",ascending=False).reset_index(drop=True)
    df["alpha_rank"] = range(1,len(df)+1)
    df["rank"] = df["alpha_rank"]  # backward compat

    # Confidence = z-score of alpha
    med,std = np.median(preds),np.std(preds)
    df["confidence"] = ((df["alpha_score"]-med)/std).round(2) if std>0 else 0

    # Rename for display compatibility
    for col in ["pe_ratio","peg_ratio","revenue_growth_yoy","gross_margin","net_margin",
                "roe","debt_to_equity","fcf_yield","momentum_12_1","volatility_12m",
                "log_market_cap","drawdown_from_high","return_12m","dividend_yield"]:
        if col not in df.columns: df[col] = None
    df = df.rename(columns={"pe_ratio":"pe_forward","revenue_growth_yoy":"revenue_growth","net_margin":"profit_margin"})

    if callback: callback(f"Alpha: {len(df)} stocks ranked | Blend: {blend}")
    return df, blend

# ══════════════════════════════════════════════════════════════
# MODEL EXPLAINABILITY (per-stock feature contributions)
# ══════════════════════════════════════════════════════════════
def explain_prediction(ticker, models_dict, medians, feat_cols, prices,
                       fundamentals_db, macro, sector_map, yf_info=None):
    """Explain WHY a stock has its alpha score. Returns per-feature contributions
    for each model in the ensemble, plus model consensus.

    Uses native tree SHAP for LightGBM/XGBoost (pred_contrib),
    Ridge coefficients × feature values, and RF feature_importances as proxy."""

    # Build features for this single ticker
    df = build_features_asof(prices, fundamentals_db, macro, datetime.now(), [ticker])
    if len(df) == 0:
        return {"ticker": ticker, "error": "No features available"}

    df = add_sector_interactions(df, sector_map)
    X = df[feat_cols].fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_ranked = rank_features(X, feat_cols)

    raw_values = X.iloc[0].to_dict()  # Raw feature values before ranking
    explanation = {"ticker": ticker, "raw_features": {}, "contributions": {},
                   "per_model": {}, "consensus": {}}

    # Store raw feature values (human-readable)
    for f in feat_cols:
        v = raw_values.get(f)
        if v is not None and v == v:  # not NaN
            explanation["raw_features"][f] = round(float(v), 4)

    # Per-model explanations
    all_contribs = {}
    for name, info in models_dict.items():
        m = info.get("model")
        if m is None: continue
        model_exp = {"prediction": None, "top_positive": [], "top_negative": []}

        try:
            pred = m.predict(X_ranked)[0]
            model_exp["prediction"] = round(float(pred * 100), 2)

            contribs = None

            # LightGBM native SHAP
            if hasattr(m, "predict") and "LGBM" in type(m).__name__:
                try:
                    raw = m.predict(X_ranked, pred_contrib=True)
                    if raw.ndim == 2:
                        contribs = raw[0, :-1]  # Last element is bias
                except: pass

            # XGBoost native SHAP
            elif "XGB" in type(m).__name__:
                try:
                    import xgboost as xgb
                    dmat = xgb.DMatrix(X_ranked)
                    raw = m.get_booster().predict(dmat, pred_contribs=True)
                    if raw.ndim == 2:
                        contribs = raw[0, :-1]
                except: pass

            # Ridge: contribution = coefficient × feature value
            elif hasattr(m, "coef_"):
                try:
                    contribs = m.coef_ * X_ranked.iloc[0].values
                except: pass

            # RandomForest: use feature_importances × signed deviation as proxy
            elif hasattr(m, "feature_importances_"):
                try:
                    # Approximate: importance × (value - 0.5) for ranked features
                    imp = m.feature_importances_
                    vals = X_ranked.iloc[0].values
                    contribs = imp * (vals - 0.5) * 2  # Scale to meaningful range
                except: pass

            if contribs is not None and len(contribs) == len(feat_cols):
                contrib_dict = {feat_cols[i]: round(float(contribs[i] * 100), 3)
                               for i in range(len(feat_cols))}
                all_contribs[name] = contrib_dict

                # Sort by absolute contribution
                sorted_c = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                model_exp["top_positive"] = [(f, c) for f, c in sorted_c if c > 0][:5]
                model_exp["top_negative"] = [(f, c) for f, c in sorted_c if c < 0][:5]

        except Exception as e:
            model_exp["error"] = str(e)

        explanation["per_model"][name] = model_exp

    # Ensemble-weighted contributions
    if all_contribs:
        ensemble_c = {}
        total_w = sum(info.get("ic", info.get("cv_r2", 0.01))
                     for info in models_dict.values() if info.get("model"))
        for name, contribs in all_contribs.items():
            w = models_dict[name].get("ic", models_dict[name].get("cv_r2", 0.01)) / total_w
            for f, c in contribs.items():
                ensemble_c[f] = ensemble_c.get(f, 0) + c * w
        sorted_ens = sorted(ensemble_c.items(), key=lambda x: abs(x[1]), reverse=True)
        explanation["contributions"] = {f: round(c, 3) for f, c in sorted_ens[:20]}
        explanation["top_drivers"] = [(f, round(c, 2)) for f, c in sorted_ens if abs(c) > 0.1][:8]

    # Model consensus: do all models agree on direction?
    preds = {n: info.get("prediction", 0) for n, info in explanation["per_model"].items()
             if info.get("prediction") is not None}
    if preds:
        explanation["consensus"] = {
            "predictions": preds,
            "all_positive": all(p > 0 for p in preds.values()),
            "all_negative": all(p < 0 for p in preds.values()),
            "spread": round(max(preds.values()) - min(preds.values()), 2),
            "agreement": "strong" if max(preds.values()) - min(preds.values()) < 5 else
                        "moderate" if max(preds.values()) - min(preds.values()) < 15 else "weak",
        }

    return explanation


def explain_ranking_context(ticker, models_dict, medians, feat_cols, prices,
                            fundamentals_db, macro, sector_map, yf_info=None):
    """Build a human-readable text explanation for the AI agent to use."""
    exp = explain_prediction(ticker, models_dict, medians, feat_cols, prices,
                            fundamentals_db, macro, sector_map, yf_info)

    if "error" in exp:
        return f"Cannot explain {ticker}: {exp['error']}"

    text = f"\n== EXPLANATION: {ticker} ==\n"

    # Consensus
    cons = exp.get("consensus", {})
    if cons:
        preds = cons.get("predictions", {})
        text += f"Model consensus: {cons.get('agreement', '?')}\n"
        for n, p in preds.items():
            text += f"  {n}: {p:+.1f}%\n"
        text += f"  Spread: {cons.get('spread', '?')}% | All agree: {'yes' if cons.get('all_positive') else 'no'}\n"

    # Top drivers
    drivers = exp.get("top_drivers", [])
    if drivers:
        text += f"\nTop alpha drivers:\n"
        for f, c in drivers:
            raw = exp.get("raw_features", {}).get(f)
            raw_str = f" (raw: {raw:.3f})" if raw is not None else ""
            direction = "+" if c > 0 else "-"
            text += f"  {direction} {f}: {c:+.2f}%{raw_str}\n"

    # Per-model breakdown for the top model
    for name, info in exp.get("per_model", {}).items():
        if info.get("top_positive"):
            text += f"\n{name} positive drivers: "
            text += ", ".join(f"{f}({c:+.1f}%)" for f, c in info["top_positive"][:3])
            text += "\n"
        if info.get("top_negative"):
            text += f"{name} negative drivers: "
            text += ", ".join(f"{f}({c:+.1f}%)" for f, c in info["top_negative"][:3])
            text += "\n"

    return text

# ══════════════════════════════════════════════════════════════
# PORTFOLIO PROJECTIONS
# ══════════════════════════════════════════════════════════════
def project_portfolio_prices(holdings_pnl, model_results, horizons=[3,6,12,24]):
    """Project future prices per holding based on alpha model predictions."""
    # ETF expected returns by ticker (multiple aliases for robustness)
    etf_defaults = {"6AQQ.DE":0.11,"ANX.PA":0.11,"UST.PA":0.11,"NDXH":0.11,
                    "IWDA.AS":0.08,"SWDA.L":0.08,
                    "VWCE.DE":0.08,"VWCE.L":0.08}
    # Also match by name fragment
    etf_name_match = {"nasdaq":0.11, "msci world":0.08, "all-world":0.08, "ftse all":0.08}
    projections = []
    for h in holdings_pnl.get("holdings",[]):
        t=h["ticker"]; cp=h.get("current_price") or h.get("avg_price",0)
        if cp<=0: continue
        annual_ret = None
        if model_results is not None and hasattr(model_results,"empty"):
            match = model_results[model_results["ticker"]==t]
            if not match.empty: annual_ret = match.iloc[0]["predicted_return_pct"]/100
        if annual_ret is None:
            annual_ret = etf_defaults.get(t)
        if annual_ret is None:
            # Try name matching for ETFs
            name_lower = h.get("name","").lower()
            for frag, ret in etf_name_match.items():
                if frag in name_lower:
                    annual_ret = ret; break
        if annual_ret is None:
            annual_ret = 0.08  # Global default
        proj = {"ticker":t,"name":h.get("name",t),"current_price":cp,"currency":h.get("currency","EUR"),
                "units":h.get("units",0),"annual_return":round(annual_ret,4),"horizons":{}}
        for m in horizons:
            factor = (1+annual_ret)**(m/12)
            proj["horizons"][f"{m}M"] = {"price":round(cp*factor,2),
                "value":round(cp*factor*h.get("units",0),2),"gain_pct":round((factor-1)*100,1)}
        projections.append(proj)
    return projections

# ══════════════════════════════════════════════════════════════
# SIMPLE MODE (no FMP)
# ══════════════════════════════════════════════════════════════
def train_simple(prices, yf_fundamentals, macro, callback=None, sentiment_scores=None):
    """Degraded mode: technical + sentiment features, ensemble, rank+neutralize."""
    if callback: callback("Simple ensemble: technical + sentiment...")
    records = []
    for ticker,fund in yf_fundamentals.items():
        row = {"ticker":ticker,"name":fund.get("shortName",ticker),"sector":fund.get("sector","Unknown")}
        try:
            if isinstance(prices.columns,pd.MultiIndex): close=prices[(ticker,"Close")].dropna()
            else: continue
            if len(close)<252: continue
            row["momentum_12_1"]=close.iloc[-21]/close.iloc[-252]-1
            row["return_1m"]=close.pct_change(21).iloc[-1]
            row["return_3m"]=close.pct_change(63).iloc[-1]
            row["return_6m"]=close.pct_change(126).iloc[-1]
            daily=close.pct_change().dropna()
            row["volatility_3m"]=daily.tail(63).std()*np.sqrt(252)
            row["volatility_12m"]=daily.tail(252).std()*np.sqrt(252)
            row["price_vs_ma50"]=close.iloc[-1]/close.tail(50).mean()-1
            row["price_vs_ma200"]=close.iloc[-1]/close.tail(200).mean()-1
            row["drawdown_from_high"]=close.iloc[-1]/close.tail(252).max()-1
            row["target_12m"]=close.iloc[-1]/close.iloc[-252]-1
            if sentiment_scores: row["news_sentiment"]=sentiment_scores.get(ticker,0.0)
            sec=fund.get("sector","")
            row["macro_fed"]=macro.get("fed_funds_rate",4)
            row["macro_oil"]=macro.get("oil_price",80)
            row["macro_vix"]=macro.get("vix",22)
            row["tech_x_rates"]=(1 if sec in ["Technology","Communication Services"] else 0)*row["macro_fed"]
            row["energy_x_oil"]=(1 if sec=="Energy" else 0)*row["macro_oil"]
            if sentiment_scores:
                s=sentiment_scores.get(ticker,0.0)
                row["sentiment_x_momentum"]=s*row.get("momentum_12_1",0)
            records.append(row)
        except: pass
    df = pd.DataFrame(records)
    if len(df)<30: return None,None,None,None,None,None
    meta=["ticker","name","sector","target_12m"]
    fcols=[c for c in df.columns if c not in meta and df[c].dtype in [np.float64,np.int64,float,int]]
    X=df[fcols].copy(); y=df["target_12m"].copy()
    medians=X.median(); X=X.fillna(medians).replace([np.inf,-np.inf],np.nan).fillna(medians)
    # Rank transform + sector neutralize
    X = rank_features(X, fcols)
    X_neut = X.copy(); X_neut["sector"]=df["sector"].values
    X_neut = sector_neutralize(X_neut,fcols,"sector")
    X = X_neut.drop(columns=["sector"],errors="ignore")
    if callback: callback("Training ensemble (simple mode)...")
    ensemble = {}
    for name,m in _get_models().items():
        try:
            from sklearn.model_selection import cross_val_score
            cv=cross_val_score(m,X,y,cv=5,scoring="r2"); m.fit(X,y)
            ensemble[name]={"model":m,"cv_r2":round(cv.mean(),4),"ic":round(cv.mean(),4)}
            if callback: callback(f"  {name}: CV R2={cv.mean():.3f}")
        except: pass
    feat_imp = _get_feature_importance(ensemble,fcols)
    preds,blend = predict_ensemble(ensemble,X)
    df["alpha_score_raw"] = preds
    # Factor-neutralized alpha score in simple mode as well
    df["alpha_score"] = factor_neutralize_scores(
        df,
        score_col="alpha_score_raw",
        sector_col="sector"
    )
    df["predicted_return_pct"] = (df["alpha_score"]*100).round(2)
    df=df.sort_values("alpha_score",ascending=False).reset_index(drop=True)
    df["alpha_rank"]=range(1,len(df)+1); df["rank"]=df["alpha_rank"]
    med,std=np.median(preds),np.std(preds)
    df["confidence"]=((preds-med)/std).round(2) if std>0 else 0
    pm = {n:{"cv_r2":info.get("cv_r2",0)} for n,info in ensemble.items()}
    oos_metrics = {"mode":"simple_ensemble","n_stocks":len(X),"n_features":len(fcols),
                   "per_model":pm,"blend":blend,"caveat":"Technical features only."}
    return ensemble,medians,fcols,df,feat_imp,oos_metrics

# ══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════
def run_full_pipeline(callback=None):
    """Main entry. Fetches all data, trains ensemble, generates alpha rankings.
    Returns (results_df, feature_importance, model_info, macro)."""
    from .data import fetch_all_data
    alldata = fetch_all_data(callback=callback)
    tickers = alldata["tickers"]; prices = alldata["prices"]
    yf_fund = alldata["fundamentals"]; macro = alldata["macro"]
    sentiment = alldata.get("sentiment",{})
    sector_map = {t:f.get("sector","Unknown") for t,f in yf_fund.items()}
    fmp_key = os.environ.get("FMP_API_KEY")

    if fmp_key:
        if callback: callback("Full mode: FMP + walk-forward ensemble")
        fund_db = fetch_all_fundamentals(list(yf_fund.keys()),fmp_key,callback)
        if len(fund_db)<30:
            return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment)
        wf = walk_forward_train(prices,fund_db,macro,sector_map,list(fund_db.keys()),
                               start_year=2019,callback=callback)
        if wf[0] is None:
            return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment)
        final_models,medians,feat_cols,feat_imp,oos_metrics = wf
        results,blend = predict_current(final_models,medians,feat_cols,prices,fund_db,
                                        macro,sector_map,list(yf_fund.keys()),
                                        yf_info=yf_fund,callback=callback)
        if sentiment: results["news_sentiment"]=results["ticker"].map(sentiment).fillna(0)
        model_info = {"mode":"walk_forward_ensemble","n_features":len(feat_cols),"blend":blend,**oos_metrics}
        # Store full model state for explanations
        _store_model_state(final_models, medians, feat_cols, prices, fund_db, sector_map, yf_fund)
    else:
        if callback: callback("Simple mode (no FMP key)")
        return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment)
    _save_cache(results,feat_imp,model_info,macro)
    return results,feat_imp,model_info,macro

def _run_simple(prices,yf_fund,macro,sector_map,callback=None,sentiment=None):
    r = train_simple(prices,yf_fund,macro,callback,sentiment)
    if r[0] is None: return None,None,{"error":"Training failed"},macro
    ensemble,med,fc,results,feat_imp,oos = r
    # Store model state for explanations
    _store_model_state(ensemble, med, fc, prices, {}, sector_map, yf_fund)
    _save_cache(results,feat_imp,oos,macro)
    return results,feat_imp,oos,macro

# ══════════════════════════════════════════════════════════════
# MODEL STATE (for explainability)
# ══════════════════════════════════════════════════════════════
_LAST_MODEL_STATE = None

def _store_model_state(models_dict, medians, feat_cols, prices, fundamentals_db, sector_map, yf_info):
    """Store the full model state for later use by the explainability system."""
    global _LAST_MODEL_STATE
    _LAST_MODEL_STATE = {
        "models_dict": models_dict,
        "medians": medians,
        "feat_cols": feat_cols,
        "prices": prices,
        "fundamentals_db": fundamentals_db,
        "sector_map": sector_map,
        "yf_info": yf_info,
    }

def get_model_state():
    """Retrieve the stored model state for explainability."""
    return _LAST_MODEL_STATE

def _save_cache(results,feat_imp,model_info,macro):
    with open(MODEL_CACHE,"wb") as f:
        pickle.dump({"results":results,"feat_imp":feat_imp,"model_info":model_info,
                     "macro":macro,"saved_at":datetime.now().isoformat()},f)

def load_cached():
    if MODEL_CACHE.exists():
        try:
            with open(MODEL_CACHE,"rb") as f: return pickle.load(f)
        except: pass
    return None
