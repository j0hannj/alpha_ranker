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
import logging
import os, json, pickle, random, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

GLOBAL_SEED = 42


def set_global_seed(seed=None):
    """Set all random seeds for reproducibility. Call at start of pipeline when deterministic_mode is True."""
    s = seed if seed is not None else GLOBAL_SEED
    np.random.seed(s)
    random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# New modular imports (refactored architecture)
from features.fundamental_features import get_fundamentals_asof, build_features_asof
from features.price_features import compute_forward_return
from features.cross_sectional import (
    rank_features,
    sector_neutralize,
    factor_neutralize_scores,
    add_sector_interactions,
    decorrelate_features,
)

logger = logging.getLogger(__name__)
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
    """Fetch FMP fundamentals. Uses per-ticker SQLite cache: only requests tickers not already cached."""
    data = {}
    to_fetch = []
    try:
        from .api_cache import get as cache_get, set as cache_set
        for t in tickers:
            cached = cache_get("fmp_fund", t, max_age_hours=7*24)
            if cached and isinstance(cached, list) and len(cached) > 0:
                data[t] = cached
            else:
                to_fetch.append(t)
    except Exception:
        to_fetch = list(tickers)
    if not to_fetch and data:
        if callback: callback(f"FMP cache (SQLite): {len(data)} tickers (no request)")
        return data
    if FUNDAMENTALS_CACHE.exists() and not data:
        try:
            cache = json.loads(FUNDAMENTALS_CACHE.read_text(encoding="utf-8"))
            if (datetime.now()-datetime.fromisoformat(cache.get("_date","2000-01-01"))).days < 7:
                data = {k: v for k, v in cache.items() if k != "_date" and k in tickers}
                to_fetch = [t for t in tickers if t not in data]
                if not to_fetch and len(data) > 50:
                    if callback: callback(f"FMP cache (file): {len(data)} tickers")
                    return data
        except Exception:
            pass
    if to_fetch and callback:
        callback(f"FMP: fetching {len(to_fetch)}/{len(tickers)} tickers (rest from cache)...")
    for i, t in enumerate(to_fetch):
        if callback and (i+1) % 25 == 0:
            callback(f"FMP: {i+1}/{len(to_fetch)}...")
        try:
            rows = fetch_fmp_quarterly(t, api_key)
            if rows:
                data[t] = rows
                try:
                    cache_set("fmp_fund", t, rows)
                except Exception:
                    pass
        except Exception:
            pass
    if data:
        try:
            payload = {**data, "_date": datetime.now().isoformat()}
            FUNDAMENTALS_CACHE.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception:
            pass
    if callback: callback(f"FMP: {len(data)} tickers loaded")
    return data

# ══════════════════════════════════════════════════════════════
# TCN / LSTM (optional PyTorch)
# ══════════════════════════════════════════════════════════════
def _make_tcn_regressor(n_features, kernel_size=3, channels=(32, 32), dropout=0.2, dilation_levels=4, device=None):
    """Sklearn-like TCN regressor using PyTorch. Input 2D (n_samples, n_features)."""
    import torch
    import torch.nn as nn

    class _TCNBlock(nn.Module):
        def __init__(self, c_in, c_out, k, dilation):
            super().__init__()
            pad = (k - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            out = self.conv(x)
            if out.size(-1) != x.size(-1):
                out = out[..., :x.size(-1)]
            return self.drop(self.act(out))

    class _TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ch = [1] + list(channels)
            for i in range(len(ch) - 1):
                for d in range(dilation_levels):
                    layers.append(_TCNBlock(ch[i] if d == 0 else ch[i + 1], ch[i + 1], kernel_size, 2 ** d))
            self.seq = nn.Sequential(*layers)
            self.fc = nn.Linear(channels[-1] * n_features, 1)

        def forward(self, x):
            # x: (batch, 1, n_features)
            h = self.seq(x)  # (batch, channels[-1], n_features)
            h = h.reshape(h.size(0), -1)
            return self.fc(h).squeeze(-1)

    class TCNRegressor:
        def __init__(self, n_features, kernel_size=3, channels=(32, 32), dropout=0.2, dilation_levels=4, epochs=50, lr=1e-3, device=None):
            self.n_features = n_features
            self.kernel_size = kernel_size
            self.channels = channels
            self.dropout = dropout
            self.dilation_levels = dilation_levels
            self.epochs = epochs
            self.lr = lr
            self.net = None
            if device == "cpu":
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def fit(self, X, y):
            if hasattr(X, "values"):
                X = X.values
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            n_features = X.shape[1]
            self.net = _TCN().to(self.device)
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            Xt = torch.from_numpy(X).reshape(-1, 1, n_features).to(self.device)
            yt = torch.from_numpy(y).to(self.device)
            self.net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                out = self.net(Xt)
                loss = ((out - yt) ** 2).mean()
                loss.backward()
                opt.step()
            return self

        def predict(self, X):
            if self.net is None:
                return np.zeros(len(X))
            if hasattr(X, "values"):
                X = X.values
            X = np.asarray(X, dtype=np.float32)
            Xt = torch.from_numpy(X).reshape(-1, 1, X.shape[1]).to(self.device)
            self.net.eval()
            with torch.no_grad():
                out = self.net(Xt)
            return out.cpu().numpy()

    return TCNRegressor(n_features, kernel_size, channels, dropout, dilation_levels, device=device)


def _make_lstm_regressor(n_features, units=64, dropout=0.2, epochs=50, lr=1e-3, device=None):
    """Sklearn-like LSTM regressor using PyTorch. Input 2D (n_samples, n_features)."""
    import torch
    import torch.nn as nn

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, units, batch_first=True, dropout=0)
            self.drop = nn.Dropout(dropout if dropout else 0)
            self.fc = nn.Linear(units, 1)

        def forward(self, x):
            # x: (batch, 1, n_features)
            out, _ = self.lstm(x)
            out = self.drop(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    class LSTMRegressor:
        def __init__(self, n_features, units=64, dropout=0.2, epochs=50, lr=1e-3, device=None):
            self.n_features = n_features
            self.units = units
            self.dropout = dropout
            self.epochs = epochs
            self.lr = lr
            self.net = None
            if device == "cpu":
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def fit(self, X, y):
            if hasattr(X, "values"):
                X = X.values
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            n_features = X.shape[1]
            self.net = _LSTM().to(self.device)
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            Xt = torch.from_numpy(X).reshape(-1, 1, n_features).to(self.device)
            yt = torch.from_numpy(y).to(self.device)
            self.net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                out = self.net(Xt)
                loss = ((out - yt) ** 2).mean()
                loss.backward()
                opt.step()
            return self

        def predict(self, X):
            if self.net is None:
                return np.zeros(len(X))
            if hasattr(X, "values"):
                X = X.values
            X = np.asarray(X, dtype=np.float32)
            Xt = torch.from_numpy(X).reshape(-1, 1, X.shape[1]).to(self.device)
            self.net.eval()
            with torch.no_grad():
                out = self.net(Xt)
            return out.cpu().numpy()

    return LSTMRegressor(n_features, units, dropout, epochs, lr, device=device)


def _get_models(config=None):
    """Build model dict. If config provided, only enabled_models are included. deterministic_mode -> n_jobs=1 for reproducibility."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    n_est = 500; depth = 5; lr = 0.03
    ridge_alpha = 10.0
    det = config.get("deterministic_mode", False) if config else False
    n_jobs = 1 if det else -1
    if config:
        n_est = config.get("n_estimators", n_est)
        depth = config.get("max_depth", depth)
        lr = config.get("learning_rate", lr)
        ridge_alpha = config.get("ridge_alpha", ridge_alpha)
    lgb_kw = dict(n_estimators=n_est,max_depth=depth,learning_rate=lr,
        subsample=0.7,colsample_bytree=0.6,min_child_samples=15,
        reg_alpha=0.3,reg_lambda=0.3,random_state=42,verbose=-1,n_jobs=n_jobs)
    if det:
        lgb_kw["deterministic"] = True
    all_models = {
        "LightGBM": LGBMRegressor(**lgb_kw),
        "XGBoost": XGBRegressor(n_estimators=n_est,max_depth=depth,learning_rate=lr,
            subsample=0.7,colsample_bytree=0.6,min_child_weight=15,
            reg_alpha=0.3,reg_lambda=0.3,random_state=42,verbosity=0,n_jobs=n_jobs),
        "Ridge": Ridge(alpha=ridge_alpha),
        "ElasticNet": ElasticNet(alpha=config.get("elastic_net_alpha",1.0) if config else 1.0,
            l1_ratio=config.get("elastic_net_l1_ratio",0.5) if config else 0.5),
        "RandomForest": RandomForestRegressor(n_estimators=min(300,n_est),max_depth=min(8,depth),
            min_samples_leaf=15,max_features=0.6,random_state=42,n_jobs=n_jobs),
    }
    # TCN / LSTM: require PyTorch and n_features (set at first fit in walk_forward)
    _torch_available = False
    try:
        import torch
        _torch_available = True
    except Exception:
        pass
    if _torch_available:
        # Placeholder instances; real n_features set when fitting. We use lazy wrappers that build net on first fit.
        tcn_k = config.get("tcn_kernel_size", 3) if config else 3
        tcn_ch = config.get("tcn_channels", [32, 32]) if config else [32, 32]
        tcn_drop = config.get("tcn_dropout", 0.2) if config else 0.2
        tcn_dil = config.get("tcn_dilation_levels", 4) if config else 4
        lstm_u = config.get("lstm_units", 64) if config else 64
        lstm_drop = config.get("lstm_dropout", 0.2) if config else 0.2

        use_gpu = (config or {}).get("use_gpu", True)
        dl_device = None if use_gpu else "cpu"
        class _LazyTCN:
            def __init__(self, dev): self._model = None; self._device = dev
            def fit(self, X, y):
                nf = X.shape[1] if hasattr(X, "shape") else len(X.columns)
                self._model = _make_tcn_regressor(nf, tcn_k, tuple(tcn_ch), tcn_drop, tcn_dil, device=self._device)
                self._model.fit(X, y)
                return self
            def predict(self, X): return self._model.predict(X) if self._model else np.zeros(len(X))
        class _LazyLSTM:
            def __init__(self, dev): self._model = None; self._device = dev
            def fit(self, X, y):
                nf = X.shape[1] if hasattr(X, "shape") else len(X.columns)
                self._model = _make_lstm_regressor(nf, lstm_u, lstm_drop, device=self._device)
                self._model.fit(X, y)
                return self
            def predict(self, X): return self._model.predict(X) if self._model else np.zeros(len(X))
        all_models["TCN"] = _LazyTCN(dl_device)
        all_models["LSTM"] = _LazyLSTM(dl_device)
    if config:
        enabled = config.get("enabled_models") or list(all_models.keys())
        return {k: v for k, v in all_models.items() if k in enabled}
    return all_models

def predict_ensemble(models_dict, X, method="ic_weighted", return_per_model=False):
    """Ensemble prediction. method: simple_average | ic_weighted_average | stacked_meta_model.
    Returns (predictions_array, blend_info_dict). If return_per_model=True, blend includes 'per_model_preds' for agreement/reliability."""
    all_preds = {}; weights = {}
    for name, info in models_dict.items():
        if info.get("model") is None: continue
        try:
            p = info["model"].predict(X)
            all_preds[name] = np.asarray(p)
            weights[name] = max(info.get("ic",info.get("cv_r2",0.01)), 0.001)
        except Exception as e:
            logger.warning("predict_ensemble: model %s predict failed: %s", name, e)
    if not all_preds: return np.zeros(len(X)), {}
    if method == "simple_average":
        final = np.mean(np.array(list(all_preds.values())), axis=0)
    else:
        total_w = sum(weights.values())
        final = sum(all_preds[n]*(weights[n]/total_w) for n in all_preds)
    blend = {n:{"weight":round(weights[n]/sum(weights.values()),3),"mean_pred":round(float(np.mean(p)),4)}
             for n,p in all_preds.items()}
    if return_per_model:
        blend["per_model_preds"] = all_preds  # name -> array of length n_samples
    return final, blend

def _apply_winsorization(X, feat_cols, quantile=0.02):
    """Cross-sectional winsorization to limit extreme values."""
    out = X.copy()
    for c in feat_cols:
        if c not in out.columns: continue
        q = out[c].quantile([quantile, 1 - quantile])
        out[c] = out[c].clip(lower=q.iloc[0], upper=q.iloc[1])
    return out

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
                       start_year=2019, horizon_months=12, callback=None, config=None, as_of_date=None):
    """Walk-forward with ensemble. Uses config from DB if not provided. as_of_date: fix date for reproducibility (default: last price date)."""
    try:
        from core.engine_config import get_model_settings, get_feature_settings, get_enabled_feature_columns
    except Exception:
        get_model_settings = get_feature_settings = get_enabled_feature_columns = None
    if config is None and get_model_settings:
        config = get_model_settings()
    if config is None:
        config = {}
    if config.get("deterministic_mode", False):
        set_global_seed(config.get("global_seed", GLOBAL_SEED))
    try:
        from core.engine_config import MODEL_IDS
    except Exception:
        MODEL_IDS = ["LightGBM", "XGBoost", "RandomForest", "Ridge", "ElasticNet", "TCN", "LSTM"]
    emode = config.get("execution_mode", "all")
    if emode == "all":
        config = {**config, "enabled_models": list(MODEL_IDS)}
    elif emode == "single":
        sid = config.get("single_model_id", "LightGBM")
        config = {**config, "enabled_models": [sid] if sid in (MODEL_IDS if isinstance(MODEL_IDS, list) else list(MODEL_IDS)) else ["LightGBM"]}
    H = config.get("prediction_horizon_months", horizon_months)
    ref_date = as_of_date
    if ref_date is None and prices is not None and hasattr(prices, "index") and len(prices.index) > 0:
        try:
            ref_date = prices.index[-1]
            if hasattr(ref_date, "to_pydatetime"):
                ref_date = ref_date.to_pydatetime()
            elif not isinstance(ref_date, datetime):
                ref_date = datetime(ref_date.year, getattr(ref_date, "month", 1), getattr(ref_date, "day", 1))
        except Exception:
            ref_date = datetime.now()
    if ref_date is None:
        ref_date = datetime.now()
    cutoff = ref_date - timedelta(days=H*30)
    rebal_dates = [datetime(y,m,1) for y in range(start_year, ref_date.year+1)
                   for m in [1,4,7,10] if datetime(y,m,1) < cutoff]
    if callback: callback(f"Walk-forward: {len(rebal_dates)} periods")
    all_periods = []; meta_cols = ["ticker","date","sector","name","forward_return"]
    for i,rd in enumerate(rebal_dates):
        if callback and (i+1)%4==0: callback(f"Features: period {i+1}/{len(rebal_dates)}")
        df = build_features_asof(prices,fundamentals_db,macro,rd,tickers)
        if len(df)==0: continue
        df = add_sector_interactions(df,sector_map)
        df["forward_return"] = df["ticker"].apply(lambda t: compute_forward_return(prices,t,rd,H))
        df = df.dropna(subset=["forward_return"])
        if len(df)<20: continue
        df["period_idx"]=i; all_periods.append(df)
    if not all_periods:
        if callback: callback("Not enough data"); return None,None,None,None,None
    full_df = pd.concat(all_periods,ignore_index=True)
    all_num_cols = [c for c in full_df.columns if c not in meta_cols+["period_idx"]
                    and full_df[c].dtype in [np.float64,np.int64,float,int]]
    if get_enabled_feature_columns and get_feature_settings:
        enabled_cols = set(get_enabled_feature_columns())
        feat_cols = [c for c in all_num_cols if c in enabled_cols]
        if not feat_cols:
            feat_cols = all_num_cols
    else:
        feat_cols = all_num_cols
    if callback: callback(f"{len(full_df)} obs, {len(feat_cols)} features, {full_df['period_idx'].nunique()} periods")

    models_template = _get_models(config)
    period_indices = sorted(full_df["period_idx"].unique()); min_train=8
    oos_preds=[]; per_model_oos = {n:[] for n in models_template.keys()}
    ensemble_method = config.get("ensemble_method") or "ic_weighted_average"
    winsorize = config.get("winsorization", False)
    winsorize_q = config.get("winsorize_quantile", 0.02)
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
        if winsorize:
            Xtr = _apply_winsorization(Xtr, feat_cols, winsorize_q)
            Xte = _apply_winsorization(Xte, feat_cols, winsorize_q)
        # Rank transform + sector neutralize for training
        Xtr_r = rank_features(Xtr,feat_cols)
        Xte_r = rank_features(Xte,feat_cols)
        if config.get("sector_neutralization", True) and "sector" in full_df.columns:
            Xtr_r = Xtr_r.copy(); Xtr_r["sector"] = trn["sector"].values
            Xtr_r = sector_neutralize(Xtr_r, feat_cols, "sector"); Xtr_r = Xtr_r.drop(columns=["sector"], errors="ignore")
            Xte_r = Xte_r.copy(); Xte_r["sector"] = tst["sector"].values
            Xte_r = sector_neutralize(Xte_r, feat_cols, "sector"); Xte_r = Xte_r.drop(columns=["sector"], errors="ignore")
        if config.get("feature_decorrelation"):
            method = config.get("decorrelation_method", "pca")
            var_ratio = float(config.get("pca_variance_ratio", 0.95))
            Xtr_r, _fitted = decorrelate_features(Xtr_r, feat_cols, method=method, variance_ratio=var_ratio)
            if _fitted is not None:
                Xte_r, _ = decorrelate_features(Xte_r, feat_cols, method=method, variance_ratio=var_ratio, fitted_transformer=_fitted)
        # Target winsorization (labels only): reduces impact of extreme forward returns during training.
        # Do not clamp model predictions at inference; if predictions are unrealistic, fix target construction or horizon.
        ytr = ytr.clip(ytr.quantile(0.02),ytr.quantile(0.98))
        models = _get_models(config)
        for name,m in models.items():
            try:
                m.fit(Xtr_r,ytr)
                p = m.predict(Xte_r)
                ic = stats.spearmanr(p,yte.values)[0] if len(yte)>5 else 0
                per_model_oos.setdefault(name, []).append(ic)
            except Exception:
                per_model_oos.setdefault(name, []).append(0)
        ens = {n: {"model": m, "ic": np.mean(per_model_oos.get(n,[])) or 0.01} for n, m in models.items()}
        ep,_ = predict_ensemble(ens,Xte_r, method=ensemble_method if isinstance(ensemble_method, str) else "ic_weighted_average")
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
        oos_metrics = {"n_predictions":0,"per_model_ic":{},"mean_ic":0,"ic_std":0,"ic_ir":0,"hit_rate":0,"spearman_rank_corr":0}
    if callback:
        callback(f"Ensemble OOS Rank IC: {oos_metrics.get('spearman_rank_corr','?')}")
        for n,ic in oos_metrics.get("per_model_ic",{}).items(): callback(f"  {n}: IC={ic:.4f}")

    # Final ensemble on ALL data with rank + sector neutral
    if callback: callback("Training final ensemble...")
    X_all = full_df[feat_cols].copy(); y_all = full_df["forward_return"].copy()
    med_final = X_all.median()
    X_all = X_all.fillna(med_final).replace([np.inf,-np.inf],np.nan).fillna(med_final)
    if winsorize:
        X_all = _apply_winsorization(X_all, feat_cols, winsorize_q)
    X_all = rank_features(X_all,feat_cols)
    if config.get("sector_neutralization", True) and "sector" in full_df.columns:
        X_all["sector"] = full_df["sector"].values
        X_all = sector_neutralize(X_all,feat_cols,"sector")
        X_all = X_all.drop(columns=["sector"],errors="ignore")
    # Same: target winsorization for final fit only; predictions are never clamped.
    y_all = y_all.clip(y_all.quantile(0.02),y_all.quantile(0.98))
    final_models = {}
    for name,m in _get_models(config).items():
        try:
            m.fit(X_all,y_all)
            final_models[name] = {"model":m,
                "ic":np.mean(per_model_oos.get(name,[])) or 0.01,
                "cv_r2":np.mean(per_model_oos.get(name,[])) or 0}
            if callback: callback(f"  {name}: trained, OOS IC={final_models[name]['ic']:.4f}")
        except Exception as e:
            logger.warning("walk_forward final fit: %s failed: %s", name, e)
    feat_imp = _get_feature_importance(final_models,feat_cols)
    return final_models, med_final, feat_cols, feat_imp, oos_metrics

# ══════════════════════════════════════════════════════════════
# CURRENT PREDICTIONS → ALPHA SCORE + RANK
# ══════════════════════════════════════════════════════════════
def predict_current(models_dict, medians, feat_cols, prices, fundamentals_db,
                    macro, sector_map, tickers, yf_info=None, callback=None, config=None, as_of_date=None):
    """Generate current alpha scores and ranks using the trained ensemble. as_of_date: use last price date when set for reproducibility."""
    if config is None:
        try:
            from core.engine_config import get_model_settings
            config = get_model_settings() or {}
        except Exception:
            config = {}
    ensemble_method = config.get("ensemble_method") or "ic_weighted_average"
    winsorize = config.get("winsorization", False)
    winsorize_q = config.get("winsorize_quantile", 0.02)
    ref_date = as_of_date
    if ref_date is None and prices is not None and hasattr(prices, "index") and len(prices.index) > 0:
        try:
            ref_date = prices.index[-1]
            if hasattr(ref_date, "to_pydatetime"):
                ref_date = ref_date.to_pydatetime()
            elif not isinstance(ref_date, datetime):
                ref_date = datetime(ref_date.year, getattr(ref_date, "month", 1), getattr(ref_date, "day", 1))
        except Exception:
            ref_date = datetime.now()
    if ref_date is None:
        ref_date = datetime.now()
    if callback: callback("Generating alpha scores...")
    df = build_features_asof(prices,fundamentals_db,macro,ref_date,tickers)
    df = add_sector_interactions(df,sector_map)
    if yf_info:
        df["sector"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("sector","Unknown"))
        df["name"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("shortName",t))
        df["current_price"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("currentPrice"))
        df["target_mean"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("targetMeanPrice"))
        df["recommendation"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("recommendationKey"))
        df["num_analysts"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("numberOfAnalystOpinions"))

    X = df[feat_cols].fillna(medians).replace([np.inf,-np.inf],np.nan).fillna(medians)
    if winsorize:
        X = _apply_winsorization(X, feat_cols, winsorize_q)
    X = rank_features(X, feat_cols)  # Same transform as training
    # Sector neutralization (mirror training pipeline)
    if config.get("sector_neutralization", True) and "sector" in df.columns:
        X_neut = X.copy()
        X_neut["sector"] = df["sector"].values
        X_neut = sector_neutralize(X_neut, feat_cols, "sector")
        X = X_neut.drop(columns=["sector"], errors="ignore")
    # Feature decorrelation (mirror training pipeline; re-fit on current X, TODO: reuse fitted from training)
    if config.get("feature_decorrelation"):
        method = config.get("decorrelation_method", "pca")
        var_ratio = float(config.get("pca_variance_ratio", 0.95))
        X, _ = decorrelate_features(X, feat_cols, method=method, variance_ratio=var_ratio)
    n_models = sum(1 for info in models_dict.values() if info.get("model") is not None)
    preds, blend = predict_ensemble(models_dict, X, method=ensemble_method, return_per_model=(n_models > 1))

    # Model agreement: fraction of models predicting positive return
    if n_models > 1 and "per_model_preds" in blend:
        per_model = blend["per_model_preds"]
        pos_count = np.zeros(len(X))
        for p in per_model.values():
            pos_count += (np.asarray(p) > 0).astype(float)
        df["model_agreement_score"] = (pos_count / n_models).round(3)
    else:
        df["model_agreement_score"] = 1.0

    # Raw alpha score before factor neutralization
    df["alpha_score_raw"] = preds
    df["alpha_score_raw"] = df["alpha_score_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Factor-neutralized alpha score (size / risk / momentum / sector)
    df["alpha_score"] = factor_neutralize_scores(
        df,
        score_col="alpha_score_raw",
        sector_col="sector"
    )
    df["alpha_score"] = df["alpha_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # predicted_return_pct must use raw score (return space); alpha_score is z-score for ranking only
    df["predicted_return_pct"] = (df["alpha_score_raw"] * 100).round(2)
    df["predicted_return_pct"] = df["predicted_return_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.sort_values("alpha_score", ascending=False).reset_index(drop=True)
    df["alpha_rank"] = range(1, len(df) + 1)
    df["rank"] = df["alpha_rank"]  # backward compat

    # Confidence = z-score of raw prediction (same scale as predicted_return_pct)
    med, std = np.median(preds), np.std(preds)
    df["confidence"] = ((preds - med) / std).round(2) if std > 0 else 0
    df["confidence"] = df["confidence"].fillna(0.0)
    # Reliability: combine alpha, confidence, and model agreement (prioritize high agreement)
    ar, cr = df["alpha_score"], df["confidence"]
    a_norm = (ar - ar.min()) / (ar.max() - ar.min() + 1e-9)
    c_norm = (cr - cr.min()) / (cr.max() - cr.min() + 1e-9)
    df["reliability_score"] = (0.4 * a_norm + 0.3 * c_norm + 0.3 * df["model_agreement_score"]).round(3)
    df["reliability_score"] = df["reliability_score"].fillna(0.0)
    assert df["predicted_return_pct"].isna().sum() == 0, f"NaN in predicted_return_pct: {df['predicted_return_pct'].isna().sum()}"

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
                except Exception as e:
                    logger.debug("explain_prediction: LightGBM contrib for %s: %s", name, e)

            # XGBoost native SHAP
            elif "XGB" in type(m).__name__:
                try:
                    import xgboost as xgb
                    dmat = xgb.DMatrix(X_ranked)
                    raw = m.get_booster().predict(dmat, pred_contribs=True)
                    if raw.ndim == 2:
                        contribs = raw[0, :-1]
                except Exception as e:
                    logger.debug("explain_prediction: XGBoost contrib for %s: %s", name, e)

            # Ridge: contribution = coefficient × feature value
            elif hasattr(m, "coef_"):
                try:
                    contribs = m.coef_ * X_ranked.iloc[0].values
                except Exception as e:
                    logger.debug("explain_prediction: Ridge contrib for %s: %s", name, e)

            # RandomForest: use feature_importances × signed deviation as proxy
            elif hasattr(m, "feature_importances_"):
                try:
                    # Approximate: importance × (value - 0.5) for ranked features
                    imp = m.feature_importances_
                    vals = X_ranked.iloc[0].values
                    contribs = imp * (vals - 0.5) * 2  # Scale to meaningful range
                except Exception as e:
                    logger.debug("explain_prediction: RandomForest contrib for %s: %s", name, e)

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
# Pipeline: model prediction (H-month return) → return projection → price projection → UI.
# Model output: predicted_return_pct = 100 * alpha_score, where alpha_score is the model's
# predicted SIMPLE return over the training horizon (H months). So predicted_return_pct=12
# means 12% return over H months, NOT annualized. We never interpret a 1-year prediction
# as a 2-year return; we compound explicitly: projected_price = current_price * (1+r)^(m/H).
# When all_horizon_results is provided, each display horizon uses that horizon's prediction
# when available (no extrapolation); otherwise we compound from primary horizon.
# No clamping of predicted returns: unrealistic predictions must be fixed at the model/target level.
def project_portfolio_prices(holdings_pnl, model_results, horizons=None, model_info=None, all_horizon_results=None):
    """Project future prices per holding from model predictions.
    Uses model's prediction horizon (H months) so a 12-month prediction is never used as 24-month.
    Formula: projected_price = current_price * (1 + predicted_return); for horizon m when we have
    a per-horizon prediction we use it once; else we compound: (1 + r_H)^(m/H).
    When all_horizon_results is provided (dict horizon_months -> {"results": df}), each display
    horizon m uses that horizon's predicted_return_pct when available (no extrapolation).
    Predicted returns are not clamped; fix root causes (target, horizon, scaling) if outputs are unrealistic.
    """
    if horizons is None:
        horizons = [3, 6, 12, 24, 120]
    H = 12
    if isinstance(model_info, dict) and model_info.get("prediction_horizon_months") is not None:
        H = int(model_info["prediction_horizon_months"])
    if H <= 0:
        H = 12
    etf_defaults = {"6AQQ.DE":0.11,"ANX.PA":0.11,"UST.PA":0.11,"NDXH":0.11,
                    "IWDA.AS":0.08,"SWDA.L":0.08,"VWCE.DE":0.08,"VWCE.L":0.08}
    etf_name_match = {"nasdaq":0.11, "msci world":0.08, "all-world":0.08, "ftse all":0.08}
    projections = []
    for h in holdings_pnl.get("holdings", []):
        t = h["ticker"]
        cp = h.get("current_price") or h.get("avg_price", 0)
        if cp <= 0:
            continue
        r_H = None
        from_model = False
        if model_results is not None and hasattr(model_results, "empty") and not model_results.empty:
            match = model_results[model_results["ticker"] == t]
            if not match.empty:
                raw_pct = match.iloc[0].get("predicted_return_pct")
                if raw_pct is not None and pd.notna(raw_pct):
                    r_H = float(raw_pct) / 100.0
                    from_model = True
        if r_H is None:
            r_H = etf_defaults.get(t)
        if r_H is None:
            name_lower = (h.get("name") or "").lower()
            for frag, ret in etf_name_match.items():
                if frag in name_lower:
                    r_H = ret
                    break
        if r_H is None:
            r_H = 0.08
        proj = {
            "ticker": t,
            "name": h.get("name", t),
            "current_price": cp,
            "currency": h.get("currency", "EUR"),
            "units": h.get("units", 0),
            "annual_return": round(r_H, 4),
            "horizons": {},
            "projection_trace": {
                "model_horizon_months": H,
                "model_return_pct": round(r_H * 100, 2),
                "output_interpretation": f"Model predicted {H}-month simple return (not annualized). Applied once for {H}M; compounded for longer horizons.",
                "formula": "projected_price = current_price * (1 + r_H)^(display_months / H)",
                "from_model": from_model,
                "per_horizon": {},
            },
        }
        for m in horizons:
            if m <= 0:
                continue
            label = "10Y" if m == 120 else f"{m}M"
            r_m = None
            used_horizon_match = False
            if all_horizon_results and m in all_horizon_results:
                res_m = all_horizon_results[m].get("results")
                if res_m is not None and hasattr(res_m, "empty") and not res_m.empty:
                    match_m = res_m[res_m["ticker"] == t]
                    if not match_m.empty:
                        raw_m = match_m.iloc[0].get("predicted_return_pct")
                        if raw_m is not None and pd.notna(raw_m):
                            r_m = float(raw_m) / 100.0
                            used_horizon_match = True
            if r_m is not None and used_horizon_match:
                factor = 1.0 + r_m
                formula_str = f"1 + {r_m:.4f} = {factor:.4f} (per-horizon prediction)"
            else:
                base = 1.0 + float(r_H)
                if base <= 0:
                    factor = 0.0
                else:
                    factor = base ** (m / H)
                    if isinstance(factor, complex):
                        factor = 0.0
                factor = float(factor)
                formula_str = f"(1 + {r_H:.4f})^({m}/{H}) = {factor:.4f}"
            projected_price = cp * factor
            proj["horizons"][label] = {
                "price": round(projected_price, 2),
                "value": round(projected_price * h.get("units", 0), 2),
                "gain_pct": round((factor - 1) * 100, 1),
            }
            proj["projection_trace"]["per_horizon"][label] = {
                "formula": formula_str,
                "factor": round(factor, 4),
                "projected_price": round(projected_price, 2),
            }
        projections.append(proj)
    return projections

# ══════════════════════════════════════════════════════════════
# SIMPLE MODE (no FMP)
# ══════════════════════════════════════════════════════════════
def train_simple(prices, yf_fundamentals, macro, callback=None, sentiment_scores=None):
    """Degraded mode: technical + sentiment features, ensemble, rank+neutralize."""
    try:
        from core.engine_config import get_model_settings
        cfg = get_model_settings()
        if cfg and cfg.get("deterministic_mode", False):
            set_global_seed(cfg.get("global_seed", GLOBAL_SEED))
    except Exception:
        pass
    if callback: callback("Simple ensemble: technical + sentiment...")
    records = []
    logger.info("train_simple: start (full universe=%d tickers)", len(yf_fundamentals))
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
        except Exception as e:
            logger.warning("train_simple: row build for %s failed: %s", ticker, e)
    df = pd.DataFrame(records)
    if len(df)<30:
        logger.warning("train_simple: abort (only %d rows, need >=30)", len(df))
        return None,None,None,None,None,None
    meta=["ticker","name","sector","target_12m"]
    fcols=[c for c in df.columns if c not in meta and df[c].dtype in [np.float64,np.int64,float,int]]
    X=df[fcols].copy(); y=df["target_12m"].copy()
    medians=X.median()
    X=X.fillna(medians).replace([np.inf,-np.inf],np.nan).fillna(medians)
    # Rank transform + sector neutralize
    X = rank_features(X, fcols)
    X_neut = X.copy(); X_neut["sector"]=df["sector"].values
    X_neut = sector_neutralize(X_neut,fcols,"sector")
    X = X_neut.drop(columns=["sector"],errors="ignore")
    # Sklearn Ridge/ElasticNet do not accept NaN: ensure matrix is finite
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X.isna().any().any() or np.isinf(X.to_numpy()).any():
        logger.warning("train_simple: non-finite values remain after clean (dropped to 0)")
    if callback: callback("Training ensemble (simple mode)...")
    logger.info("train_simple: universe=%d → %d rows (≥252d), %d features", len(yf_fundamentals), X.shape[0], X.shape[1])
    ensemble = {}
    for name,m in _get_models().items():
        try:
            from sklearn.model_selection import cross_val_score
            cv=cross_val_score(m,X,y,cv=5,scoring="r2"); m.fit(X,y)
            ensemble[name]={"model":m,"cv_r2":round(cv.mean(),4),"ic":round(cv.mean(),4)}
            if callback: callback(f"  {name}: CV R2={cv.mean():.3f}")
        except Exception as e:
            logger.warning("train_simple: ensemble fit %s failed: %s", name, e)
    feat_imp = _get_feature_importance(ensemble,fcols)
    preds,blend = predict_ensemble(ensemble,X)
    df["alpha_score_raw"] = preds
    df["alpha_score_raw"] = df["alpha_score_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["alpha_score"] = factor_neutralize_scores(
        df,
        score_col="alpha_score_raw",
        sector_col="sector"
    )
    df["alpha_score"] = df["alpha_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["predicted_return_pct"] = (df["alpha_score_raw"] * 100).round(2)
    df["predicted_return_pct"] = df["predicted_return_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["model_agreement_score"] = 1.0
    df=df.sort_values("alpha_score",ascending=False).reset_index(drop=True)
    df["alpha_rank"]=range(1,len(df)+1); df["rank"]=df["alpha_rank"]
    med,std=np.median(preds),np.std(preds)
    df["confidence"]=((preds-med)/std).round(2) if std>0 else 0
    df["confidence"] = df["confidence"].fillna(0.0)
    ar, cr = df["alpha_score"], df["confidence"]
    a_norm = (ar - ar.min()) / (ar.max() - ar.min() + 1e-9)
    c_norm = (cr - cr.min()) / (cr.max() - cr.min() + 1e-9)
    df["reliability_score"] = (0.4 * a_norm + 0.3 * c_norm + 0.3 * df["model_agreement_score"]).round(3)
    df["reliability_score"] = df["reliability_score"].fillna(0.0)
    assert df["predicted_return_pct"].isna().sum() == 0, f"NaN in predicted_return_pct: {df['predicted_return_pct'].isna().sum()}"
    pm = {n:{"cv_r2":info.get("cv_r2",0)} for n,info in ensemble.items()}
    oos_metrics = {"mode":"simple_ensemble","n_stocks":len(X),"n_features":len(fcols),
                   "per_model":pm,"blend":blend,"caveat":"Technical features only.",
                   "mean_ic":0,"ic_ir":0,"hit_rate":0.5,"spearman_rank_corr":0}
    return ensemble,medians,fcols,df,feat_imp,oos_metrics

# ══════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ══════════════════════════════════════════════════════════════
def detect_market_regime(prices, macro=None, regime_config=None):
    """Regime from 6m return and recent vol. Thresholds from regime_config (e.g. risk_settings).
    Returns dict: regime (str), return_6m, volatility_21d, vix (if macro)."""
    out = {"regime": "unknown", "return_6m": None, "volatility_21d": None, "vix": None}
    rc = regime_config or {}
    bull_th = float(rc.get("bull_return_6m", 0.05))
    bear_th = float(rc.get("bear_return_6m", -0.05))
    high_vol = float(rc.get("high_vol_threshold", 0.25))
    low_vol = float(rc.get("low_vol_threshold", 0.15))
    if prices is None or not hasattr(prices, "columns"):
        return out
    try:
        if isinstance(prices.columns, pd.MultiIndex):
            cols = [c for c in prices.columns if isinstance(c, tuple) and len(c) == 2]
            if not cols:
                return out
            ticker = cols[0][0]
            close = prices[(ticker, "Close")].dropna() if (ticker, "Close") in prices.columns else None
        else:
            close = prices["Close"].dropna() if "Close" in prices.columns else prices.iloc[:, 0].dropna()
        if close is None or len(close) < 126:
            return out
        ret_6m = (close.iloc[-1] / close.iloc[-126]) - 1 if len(close) >= 126 else None
        daily = close.pct_change().dropna()
        vol_21 = float(daily.tail(21).std() * np.sqrt(252)) if len(daily) >= 21 else None
        out["return_6m"] = round(float(ret_6m), 4) if ret_6m is not None else None
        out["volatility_21d"] = round(vol_21, 4) if vol_21 is not None else None
        if isinstance(macro, dict) and macro.get("vix") is not None:
            out["vix"] = float(macro["vix"])
        if ret_6m is not None:
            out["regime"] = "bull" if ret_6m > bull_th else "bear" if ret_6m < bear_th else "neutral"
        if vol_21 is not None:
            vol_label = "high_vol" if vol_21 > high_vol else "low_vol" if vol_21 < low_vol else "medium_vol"
            out["regime"] = out["regime"] + "_" + vol_label
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════════
# MODEL HEALTH (diagnostic for UI)
# ══════════════════════════════════════════════════════════════
def assess_model_health(oos_metrics):
    """Return (verdict, color, message) for display in Rankings tab. oos_metrics = model_info from run."""
    if not oos_metrics or not isinstance(oos_metrics, dict):
        return "—", "#71717a", "Run model to see health."
    ic = oos_metrics.get("mean_ic", 0)
    icir = oos_metrics.get("ic_ir", 0)
    hit = oos_metrics.get("hit_rate", 0)
    mode = oos_metrics.get("mode", "")
    if ic is None or (isinstance(ic, float) and ic != ic):
        ic = 0
    if hit is None or (isinstance(hit, float) and hit != hit):
        hit = 0
    if icir is None or (isinstance(icir, float) and icir != icir):
        icir = 0
    if mode == "simple_ensemble":
        reason = oos_metrics.get("simple_reason", "")
        if reason == "no_fmp_key":
            msg = "Momentum only: add FMP API key in Settings → Portfolio/API keys for full model (fundamentals + multi-horizon)."
        elif reason == "fmp_few_tickers":
            msg = "Momentum only: FMP returned too few fundamentals (<30). Check API key or try again later."
        elif reason == "all_horizons_failed":
            msg = "Momentum only: all horizons failed (data/horizon). Check data range and FMP coverage."
        else:
            msg = "Momentum ranking only (no FMP or walk-forward). Add FMP key in Settings for full model."
        return "SIMPLE", "#a1a1aa", msg
    if ic > 0.05 and icir > 0.5 and hit > 0.6:
        return "STRONG", "#34d399", "Consistent predictive signal detected."
    elif ic > 0.02 and hit > 0.5:
        return "MODERATE", "#fbbf24", "Some signal — rankings may shift between runs."
    elif ic > 0:
        return "WEAK", "#fb923c", "Weak signal — use rankings with caution."
    elif oos_metrics.get("n_predictions", 0) == 0:
        return "NO OOS", "#fb923c", "Walk-forward had no OOS periods. Check data/horizon."
    else:
        return "NO SIGNAL", "#f87171", "No predictive power. Rankings are effectively random."


# ══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════
def run_full_pipeline(callback=None):
    """Main entry. Fetches all data, trains ensemble, generates alpha rankings.
    Returns (results_df, feature_importance, model_info, macro). Uses as_of_date=last price date when deterministic_mode for reproducibility.
    Data years are sized to the max horizon (e.g. 10Y horizon → 10 years of history)."""
    from .data import fetch_all_data
    try:
        from core.engine_config import get_model_settings, init_default_config
        init_default_config()
        config = get_model_settings() or {}
    except Exception:
        config = {}
    horizons_cfg = config.get("horizons", [3, 6, 12, 24, 120])
    training_window_years = int(config.get("training_window_years", 3))
    try:
        from .engine_config import get_data_settings
        max_history = int(get_data_settings().get("data_max_history_years", 15))
    except Exception:
        max_history = 15
    data_years = min(max(training_window_years, max(horizons_cfg) // 12 if horizons_cfg else 5), max_history)
    if callback: callback(f"Loading up to {data_years}y data (max {max_history}y) for horizons {horizons_cfg}...")
    alldata = fetch_all_data(years=data_years, callback=callback)
    tickers = alldata["tickers"]; prices = alldata["prices"]
    yf_fund = alldata["fundamentals"]; macro = alldata["macro"]
    sentiment = alldata.get("sentiment",{})
    # Train on full universe. ISIN is used only for display (get_display_id) when available.
    sector_map = {t:f.get("sector","Unknown") for t,f in yf_fund.items()}
    fmp_key = os.environ.get("FMP_API_KEY")
    as_of_date = None
    if prices is not None and hasattr(prices, "index") and len(prices.index) > 0:
        try:
            as_of_date = prices.index[-1]
            if hasattr(as_of_date, "to_pydatetime"):
                as_of_date = as_of_date.to_pydatetime()
            elif not isinstance(as_of_date, datetime):
                as_of_date = datetime(as_of_date.year, getattr(as_of_date, "month", 1), getattr(as_of_date, "day", 1))
        except Exception:
            pass

    if fmp_key:
        if callback: callback("Full mode: FMP + walk-forward ensemble")
        try:
            from core.engine_config import get_model_settings, init_default_config
            init_default_config()
            config = get_model_settings()
            if config and config.get("deterministic_mode", False):
                set_global_seed(config.get("global_seed", GLOBAL_SEED))
        except Exception:
            config = {}
        fund_db = fetch_all_fundamentals(list(yf_fund.keys()),fmp_key,callback)
        if len(fund_db)<30:
            return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment,alldata.get("data_freshness"), simple_reason="fmp_few_tickers")
        horizons = (config or {}).get("horizons", [3, 6, 12, 24, 120])
        primary_H = (config or {}).get("primary_horizon", 12)
        ref_year = as_of_date.year if as_of_date else datetime.now().year
        all_horizon_results = {}
        total_h = len(horizons)
        for idx, H in enumerate(horizons):
            lbl = "10Y" if H == 120 else f"{H}M"
            if callback:
                callback(f"Training {lbl} horizon ({idx+1}/{total_h})...", (idx + 0.1) / total_h)
            config_h = {**(config or {}), "prediction_horizon_months": H}
            # Calibration window matches horizon: 10Y horizon → 10 years of data
            start_year_H = ref_year - max(H // 12, 1)
            wf = walk_forward_train(prices,fund_db,macro,sector_map,list(fund_db.keys()),
                                   start_year=start_year_H,horizon_months=H,callback=callback,config=config_h,as_of_date=as_of_date)
            if wf[0] is None:
                if callback: callback(f"  {lbl}: insufficient data, skipping", (idx + 1) / total_h)
                continue
            final_models,medians,feat_cols,feat_imp,oos_metrics = wf
            if config_h.get("execution_mode") == "single":
                single_id = config_h.get("single_model_id")
                if single_id and single_id in final_models:
                    final_models = {single_id: final_models[single_id]}
            results_h,blend = predict_current(final_models,medians,feat_cols,prices,fund_db,
                                            macro,sector_map,list(yf_fund.keys()),
                                            yf_info=yf_fund,callback=callback,config=config_h,as_of_date=as_of_date)
            if sentiment: results_h["news_sentiment"] = results_h["ticker"].map(sentiment).fillna(0)
            all_horizon_results[H] = {"results": results_h,"feat_imp": feat_imp,"oos_metrics": oos_metrics,"blend": blend}
            if callback: callback(f"  {lbl} done: IC={oos_metrics.get('spearman_rank_corr','?')} | {len(results_h)} stocks", (idx + 1) / total_h)
        if not all_horizon_results:
            return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment,alldata.get("data_freshness"), simple_reason="all_horizons_failed")
        primary = all_horizon_results.get(primary_H) or next(iter(all_horizon_results.values()))
        results = primary["results"]
        feat_imp = primary["feat_imp"]
        oos_metrics = primary["oos_metrics"]
        n_f = len(primary["feat_imp"]) if isinstance(primary["feat_imp"], (list, dict)) else 0
        model_info = {"mode":"walk_forward_ensemble","n_features":n_f,"blend":primary["blend"],
                      "prediction_horizon_months": primary_H,"horizons_trained": list(all_horizon_results.keys()),"primary_horizon": primary_H,
                      "per_horizon_metrics": {H: d["oos_metrics"] for H,d in all_horizon_results.items()}, **oos_metrics}
        try:
            from core.engine_config import get_risk_settings
            regime_config = get_risk_settings()
        except Exception:
            regime_config = {}
        model_info["market_regime"] = detect_market_regime(prices, macro, regime_config=regime_config)
        _store_model_state(None, None, None, prices, fund_db, sector_map, yf_fund)
    else:
        if callback: callback("Simple mode (no FMP key)")
        return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment,alldata.get("data_freshness"), simple_reason="no_fmp_key")
    data_freshness = alldata.get("data_freshness")
    if data_freshness:
        model_info["data_freshness"] = data_freshness
    # Expose discovered_at for UI "NEW" badge (recent discoveries)
    if not results.empty and "ticker" in results.columns and alldata.get("fundamentals"):
        results["discovered_at"] = results["ticker"].map(
            lambda t: alldata["fundamentals"].get(t, {}).get("discovered_at")
        )
    # Persist ranking snapshot for stability (before cache so next run can compare)
    try:
        from core import portfolio
        portfolio.save_ranking_snapshot(results)
    except Exception:
        pass
    _save_cache(results,feat_imp,model_info,macro,all_horizons=all_horizon_results)
    return results,feat_imp,model_info,macro,all_horizon_results

def _run_simple(prices,yf_fund,macro,sector_map,callback=None,sentiment=None,data_freshness=None, simple_reason=None):
    r = train_simple(prices,yf_fund,macro,callback,sentiment)
    if r[0] is None: return None,None,{"error":"Training failed"},macro,None
    ensemble,med,fc,results,feat_imp,oos = r
    oos["prediction_horizon_months"] = 12
    oos["simple_reason"] = simple_reason
    if data_freshness: oos["data_freshness"] = data_freshness
    _store_model_state(ensemble, med, fc, prices, {}, sector_map, yf_fund)
    try:
        from core import portfolio
        portfolio.save_ranking_snapshot(results)
    except Exception:
        pass
    _save_cache(results,feat_imp,oos,macro)
    return results,feat_imp,oos,macro,None

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

def _save_cache(results,feat_imp,model_info,macro,all_horizons=None):
    saved_at = datetime.now().isoformat()
    if isinstance(model_info, dict):
        model_info = {**model_info, "saved_at": saved_at}
    payload = {"results":results,"feat_imp":feat_imp,"model_info":model_info,"macro":macro,"saved_at":saved_at}
    if all_horizons is not None:
        payload["all_horizons"] = all_horizons
    with open(MODEL_CACHE,"wb") as f:
        pickle.dump(payload,f)

def load_cached():
    if MODEL_CACHE.exists():
        try:
            with open(MODEL_CACHE,"rb") as f: return pickle.load(f)
        except Exception as e:
            logger.warning("load_cached: pickle load failed: %s", e)
    return None


def clear_model_cache():
    """Invalidate cached model results so next Run Model uses current config. Call after Apply in Settings."""
    if MODEL_CACHE.exists():
        try:
            MODEL_CACHE.unlink()
        except Exception as e:
            logger.warning("clear_model_cache: unlink failed: %s", e)
