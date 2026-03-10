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
try:
    from .features.fundamental_features import get_fundamentals_asof, build_features_asof
    from .features.price_features import compute_forward_return
    from .features.cross_sectional import (
        rank_features,
        sector_neutralize,
        factor_neutralize_scores,
        add_sector_interactions,
        decorrelate_features,
        Winsorizer,
        RankTransformer,
        FeatureDecorrelator,
        SectorNeutralizer,
    )
except ImportError:
    # Fallback when core is imported without package context
    from features.fundamental_features import get_fundamentals_asof, build_features_asof
    from features.price_features import compute_forward_return
    from features.cross_sectional import (
        rank_features,
        sector_neutralize,
        factor_neutralize_scores,
        add_sector_interactions,
        decorrelate_features,
        Winsorizer,
        RankTransformer,
        FeatureDecorrelator,
        SectorNeutralizer,
    )

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LightGBM.*")
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
        except Exception as e:
            logger.warning("fetch_fmp_quarterly: request failed for %s [%s]: %s", ticker, ep, e)
            raw[name] = []
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
            try:
                cached = cache_get("fmp_fund", t, max_age_hours=7*24)
            except Exception as e:
                logger.warning("fetch_all_fundamentals: cache get failed for %s: %s", t, e)
                cached = None
            if cached and isinstance(cached, list) and len(cached) > 0:
                data[t] = cached
            else:
                to_fetch.append(t)
    except Exception as e:
        logger.warning("fetch_all_fundamentals: api_cache unavailable, will refetch all: %s", e)
        to_fetch = list(tickers)
    if not to_fetch and data:
        if callback: callback(f"FMP cache (SQLite): {len(data)} tickers (no request)")
        return data
    if to_fetch:
        try:
            test = fetch_fmp_quarterly(to_fetch[0], api_key)
            if not test:
                raise Exception("FMP returned empty")
        except Exception as e:
            logger.warning("FMP test call failed: %s — skipping all FMP fetches", e)
            if callback:
                callback("FMP unavailable. Skipping.")
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
        except Exception as e:
            logger.warning("fetch_all_fundamentals: FUNDAMENTALS_CACHE read failed: %s", e)
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
                except Exception as e:
                    logger.warning("fetch_all_fundamentals: cache set failed for %s: %s", t, e)
        except Exception as e:
            logger.warning("fetch_all_fundamentals: fetch failed for %s: %s", t, e)
    if data:
        try:
            payload = {**data, "_date": datetime.now().isoformat()}
            FUNDAMENTALS_CACHE.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("fetch_all_fundamentals: FUNDAMENTALS_CACHE write failed: %s", e)
    if callback: callback(f"FMP: {len(data)} tickers loaded")
    return data


def build_fundamentals_from_yfinance(prices, yf_fund, callback=None):
    """
    Build fundamentals_db in FMP-like format from yfinance.
    Used when FMP returns 403. Slower (~1–2s per ticker) but works; results cached in api_cache (yf_fund, 7 days).
    """
    import time
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("build_fundamentals_from_yfinance: yfinance not installed")
        return {}
    fund_db = {}
    tickers_with_prices = []
    if prices is not None and isinstance(prices.columns, pd.MultiIndex):
        available = set(prices.columns.get_level_values(0).unique())
        tickers_with_prices = [t for t in yf_fund.keys() if t in available]
    else:
        tickers_with_prices = list(yf_fund.keys())
    max_tickers = 500
    if len(tickers_with_prices) > max_tickers:
        us = [t for t in tickers_with_prices if "." not in t]
        non_us = [t for t in tickers_with_prices if "." in t]
        tickers_with_prices = (us + non_us)[:max_tickers]
    if callback:
        callback(f"Yahoo fundamentals: fetching {len(tickers_with_prices)} tickers...")
    errors_in_a_row = 0
    try:
        from .api_cache import get as cache_get, set as cache_set
    except Exception:
        cache_get = cache_set = None
    for i, ticker in enumerate(tickers_with_prices):
        if callback and (i + 1) % 50 == 0:
            callback(f"  Yahoo fundamentals: {i+1}/{len(tickers_with_prices)}...")
        try:
            if cache_get:
                try:
                    cached = cache_get("yf_fund", ticker, max_age_hours=7 * 24)
                    if cached and isinstance(cached, list) and len(cached) > 0:
                        fund_db[ticker] = cached
                        errors_in_a_row = 0
                        continue
                except Exception:
                    pass
            t = yf.Ticker(ticker)
            quarterly_records = []
            try:
                inc = t.quarterly_financials
                bs = t.quarterly_balance_sheet
                cf = t.quarterly_cashflow
                info = t.info or {}
                if inc is not None and not inc.empty:
                    for col_date in inc.columns:
                        if hasattr(col_date, "strftime"):
                            period_date = col_date.strftime("%Y-%m-%d")
                            filing_dt = col_date + pd.Timedelta(days=45) if hasattr(col_date, "__add__") else col_date
                            filing_date_str = filing_dt.strftime("%Y-%m-%d") if hasattr(filing_dt, "strftime") else period_date
                        else:
                            period_date = str(col_date)
                            filing_date_str = period_date

                        def _get(df, *keys):
                            if df is None or df.empty:
                                return None
                            for k in keys:
                                if k in df.index and col_date in df.columns:
                                    v = df.loc[k, col_date]
                                    if pd.notna(v):
                                        return float(v)
                            return None

                        revenue = _get(inc, "Total Revenue", "Revenue")
                        net_income = _get(inc, "Net Income", "Net Income Common Stockholders")
                        ebitda = _get(inc, "EBITDA", "Normalized EBITDA")
                        eps = _get(inc, "Basic EPS", "Diluted EPS")
                        gross_profit = _get(inc, "Gross Profit")
                        operating_income = _get(inc, "Operating Income", "Operating Revenue")
                        total_equity = _get(bs, "Total Equity Gross Minority Interest", "Stockholders Equity", "Total Stockholders Equity")
                        total_debt = _get(bs, "Total Debt", "Long Term Debt")
                        total_assets = _get(bs, "Total Assets")
                        current_assets = _get(bs, "Current Assets")
                        current_liabilities = _get(bs, "Current Liabilities")
                        fcf = _get(cf, "Free Cash Flow")
                        market_cap = info.get("marketCap")
                        shares = info.get("sharesOutstanding")
                        roe_val = (net_income / total_equity * 4) if net_income and total_equity and total_equity != 0 else info.get("returnOnEquity")
                        de_val = (total_debt / total_equity) if total_debt and total_equity and total_equity != 0 else info.get("debtToEquity")
                        cr_val = (current_assets / current_liabilities) if current_assets and current_liabilities and current_liabilities != 0 else info.get("currentRatio")
                        fcf_ps = (fcf / shares) if fcf and shares and shares > 0 else None
                        rev_ps = (revenue / shares) if revenue and shares and shares > 0 else info.get("revenuePerShare")
                        gm = (gross_profit / revenue) if gross_profit and revenue and revenue > 0 else info.get("grossMargins")
                        om = (operating_income / revenue) if operating_income and revenue and revenue > 0 else info.get("operatingMargins")
                        nm = (net_income / revenue) if net_income and revenue and revenue > 0 else info.get("profitMargins")
                        record = {
                            "ticker": ticker,
                            "filing_date": filing_date_str,
                            "period_date": period_date,
                            "revenue": revenue,
                            "net_income": net_income,
                            "eps": eps,
                            "ebitda": ebitda,
                            "pe_ratio": info.get("trailingPE"),
                            "pb_ratio": info.get("priceToBook"),
                            "ev_ebitda": info.get("enterpriseToEbitda"),
                            "roe": roe_val,
                            "debt_to_equity": de_val,
                            "current_ratio": cr_val,
                            "fcf_per_share": fcf_ps,
                            "market_cap": market_cap,
                            "dividend_yield": info.get("dividendYield"),
                            "revenue_per_share": rev_ps,
                            "gross_margin": gm,
                            "operating_margin": om,
                            "net_margin": nm,
                            "peg_ratio": info.get("pegRatio"),
                        }
                        quarterly_records.append(record)
            except Exception as e:
                logger.debug("yf quarterly for %s failed: %s", ticker, e)
            if not quarterly_records:
                try:
                    info = t.info or {}
                    if info.get("marketCap"):
                        sh = info.get("sharesOutstanding")
                        fcf_ps = (info.get("freeCashflow") / sh) if info.get("freeCashflow") and sh else None
                        record = {
                            "ticker": ticker,
                            "filing_date": datetime.now().strftime("%Y-%m-%d"),
                            "period_date": datetime.now().strftime("%Y-%m-%d"),
                            "revenue": info.get("totalRevenue"),
                            "net_income": info.get("netIncomeToCommon"),
                            "eps": info.get("trailingEps"),
                            "ebitda": info.get("ebitda"),
                            "pe_ratio": info.get("trailingPE"),
                            "pb_ratio": info.get("priceToBook"),
                            "ev_ebitda": info.get("enterpriseToEbitda"),
                            "roe": info.get("returnOnEquity"),
                            "debt_to_equity": info.get("debtToEquity"),
                            "current_ratio": info.get("currentRatio"),
                            "fcf_per_share": fcf_ps,
                            "market_cap": info.get("marketCap"),
                            "dividend_yield": info.get("dividendYield"),
                            "revenue_per_share": info.get("revenuePerShare"),
                            "gross_margin": info.get("grossMargins"),
                            "operating_margin": info.get("operatingMargins"),
                            "net_margin": info.get("profitMargins"),
                            "peg_ratio": info.get("pegRatio"),
                        }
                        quarterly_records = [record]
                except Exception as e:
                    logger.debug("yf .info for %s failed: %s", ticker, e)
            if quarterly_records:
                quarterly_records.sort(key=lambda r: r.get("filing_date", ""))
                fund_db[ticker] = quarterly_records
                errors_in_a_row = 0
                if cache_set:
                    try:
                        cache_set("yf_fund", ticker, quarterly_records)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("build_fundamentals_from_yfinance: %s: %s", ticker, e)
            errors_in_a_row += 1
            if errors_in_a_row > 10:
                if callback:
                    callback("  Yahoo rate limit detected, pausing 5s...")
                time.sleep(5)
                errors_in_a_row = 0
        if (i + 1) % 20 == 0:
            time.sleep(0.5)
    if callback:
        callback(f"Yahoo fundamentals: {len(fund_db)} tickers loaded ({sum(len(v) for v in fund_db.values())} quarterly records)")
    # Alimenter la table fundamentals (append-only) comme pour les prix
    try:
        from . import portfolio as _pf
        rows = []
        for ticker, records in fund_db.items():
            yinfo = yf_fund.get(ticker, {})
            sector = yinfo.get("sector")
            industry = yinfo.get("industry")
            country = yinfo.get("country")
            exchange = yinfo.get("exchange")
            for r in records:
                as_of = r.get("period_date") or r.get("filing_date")
                if not as_of:
                    continue
                mcap = r.get("market_cap")
                fcf_ps = r.get("fcf_per_share")
                fcf_yield = None
                if fcf_ps is not None and mcap is not None and mcap > 0:
                    try:
                        fcf_yield = float(fcf_ps) * 1e6 / float(mcap)
                    except (TypeError, ValueError):
                        pass
                rows.append({
                    "ticker": ticker,
                    "as_of_date": as_of,
                    "currency": None,
                    "revenue": r.get("revenue"),
                    "net_income": r.get("net_income"),
                    "eps": r.get("eps"),
                    "ebitda": r.get("ebitda"),
                    "free_cash_flow": None,
                    "shares_outstanding": None,
                    "market_cap": mcap,
                    "pe_ratio": r.get("pe_ratio"),
                    "pb_ratio": r.get("pb_ratio"),
                    "ev_ebitda": r.get("ev_ebitda"),
                    "fcf_yield": fcf_yield,
                    "dividend_yield": r.get("dividend_yield"),
                    "roe": r.get("roe"),
                    "gross_margin": r.get("gross_margin"),
                    "operating_margin": r.get("operating_margin"),
                    "net_margin": r.get("net_margin"),
                    "debt_to_equity": r.get("debt_to_equity"),
                    "current_ratio": r.get("current_ratio"),
                    "peg_ratio": r.get("peg_ratio"),
                    "sector": sector,
                    "industry": industry,
                    "country": country,
                    "exchange": exchange,
                    "source": "yahoo",
                    "quality_flag": None,
                })
        if rows:
            _pf.upsert_fundamentals(rows)
            if callback:
                callback(f"  DB: {len(rows)} fundamental rows written")
    except Exception as e:
        logger.warning("build_fundamentals_from_yfinance: upsert_fundamentals failed: %s", e)
    return fund_db

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
        # NOTE: TCN/LSTM require sequential input (multiple timesteps per sample)
        # to be useful. Current pipeline feeds flat cross-sectional features
        # reshaped to seq_len=1, so these models add noise without value.
        # Enable only if input is restructured to include temporal sequences.
        if config.get("enable_deep_learning_models", False):
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
            ic_val = info.get("ic", info.get("cv_r2", 0.01))
            weights[name] = (ic_val if (ic_val is not None and ic_val > 0) else 0.0)
        except Exception as e:
            logger.warning("predict_ensemble: model %s predict failed: %s", name, e)
    if not all_preds: return np.zeros(len(X)), {}
    total_w = sum(weights.values())
    if method == "simple_average" or total_w <= 0:
        final = np.mean(np.array(list(all_preds.values())), axis=0)
        total_w = total_w or 1.0  # for blend display
    else:
        final = sum(all_preds[n]*(weights[n]/total_w) for n in all_preds)
    w_sum = sum(weights.values()) or 1.0
    blend = {n:{"weight":round(weights[n]/w_sum,3),"mean_pred":round(float(np.mean(p)),4)}
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
                       start_year=2019, horizon_months=12, callback=None, config=None, as_of_date=None,
                       macro_by_region=None):
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
    cross_section_sizes = []
    try:
        from .data import get_region_for_ticker as _get_region
    except Exception:
        _get_region = None
    for i,rd in enumerate(rebal_dates):
        if callback and (i+1)%4==0: callback(f"Features: period {i+1}/{len(rebal_dates)}")
        df = build_features_asof(
            prices, fundamentals_db, macro, rd, tickers,
            macro_by_region=macro_by_region, get_region=_get_region if macro_by_region else None,
        )
        if len(df)==0: continue
        df = add_sector_interactions(df,sector_map)
        df["forward_return"] = df["ticker"].apply(lambda t: compute_forward_return(prices,t,rd,H))
        df = df.dropna(subset=["forward_return"])
        cross_section_sizes.append({"period_date": rd, "cross_section_size": len(df), "horizon_months": H})
        if len(df)<20: continue
        df["period_idx"]=i; all_periods.append(df)
    if not all_periods:
        if callback: callback("Not enough data"); return None,None,None,None,None,None,None,None,None,cross_section_sizes
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
    n_unique_tickers = full_df["ticker"].nunique()
    if callback: callback(f"{len(full_df)} obs, {n_unique_tickers} tickers, {len(feat_cols)} features, {full_df['period_idx'].nunique()} periods")

    model_names = config.get("enabled_models") or list(_get_models(config).keys())
    period_indices = sorted(full_df["period_idx"].unique())
    # Adaptive min_train: need at least 1 train period and 1 test; cap at 8 so we don't overfit with too few OOS folds
    min_train = max(1, min(8, len(period_indices) - 1)) if len(period_indices) >= 2 else 0
    oos_preds=[]; per_model_oos = {n:[] for n in model_names}
    ensemble_method = config.get("ensemble_method") or "ic_weighted_average"
    winsorize = config.get("winsorization", False)
    winsorize_q = config.get("winsorize_quantile", 0.02)
    min_trn = max(100, len(feat_cols) * 3)
    min_tst = 15
    n_periods = len(period_indices)
    if n_periods <= 4:
        min_trn = max(50, len(feat_cols) * 2)
        min_tst = 10
    for i,tp in enumerate(period_indices):
        if i<min_train: continue
        trn = full_df[full_df["period_idx"].isin(period_indices[:i])]
        tst = full_df[full_df["period_idx"]==tp]
        if len(trn)<min_trn or len(tst)<min_tst: continue
        Xtr,ytr = trn[feat_cols].copy(), trn["forward_return"].copy()
        Xte,yte = tst[feat_cols].copy(), tst["forward_return"].copy()
        med = Xtr.median()
        Xtr = Xtr.fillna(med).replace([np.inf,-np.inf],np.nan).fillna(med)
        Xte = Xte.fillna(med).replace([np.inf,-np.inf],np.nan).fillna(med)
        # Fit all transformers on training data only; transform both train and test (no test refitting).
        if winsorize:
            wiz = Winsorizer(quantile=winsorize_q)
            wiz.fit(Xtr, feat_cols)
            Xtr = wiz.transform(Xtr)
            Xte = wiz.transform(Xte)
        rank_t = RankTransformer()
        rank_t.fit(Xtr, feat_cols)
        Xtr_r = rank_t.transform(Xtr)
        Xte_r = rank_t.transform(Xte)
        if config.get("sector_neutralization", True) and "sector" in full_df.columns:
            Xtr_r = Xtr_r.copy(); Xtr_r["sector"] = trn["sector"].values
            Xte_r = Xte_r.copy(); Xte_r["sector"] = tst["sector"].values
            sn = SectorNeutralizer()
            sn.fit(Xtr_r, feat_cols, "sector")
            Xtr_r = sn.transform(Xtr_r, "sector").drop(columns=["sector"], errors="ignore")
            Xte_r = sn.transform(Xte_r, "sector").drop(columns=["sector"], errors="ignore")
        if config.get("feature_decorrelation"):
            method = config.get("decorrelation_method", "pca")
            var_ratio = float(config.get("pca_variance_ratio", 0.95))
            dec = FeatureDecorrelator(method=method, variance_ratio=var_ratio)
            dec.fit(Xtr_r, feat_cols)
            Xtr_r = dec.transform(Xtr_r)
            Xte_r = dec.transform(Xte_r)
        # Forward returns are never clipped.
        models = _get_models(config)
        for name,m in models.items():
            try:
                m.fit(Xtr_r,ytr)
                p = m.predict(Xte_r)
                ic = stats.spearmanr(p,yte.values)[0] if len(yte)>5 else 0
                per_model_oos.setdefault(name, []).append(ic)
            except Exception as e:
                logger.debug("walk_forward_train OOS: %s failed: %s", name, e)
                per_model_oos.setdefault(name, []).append(0)
        # IC-Information-Ratio weighted: penalize inconsistent models
        ens = {}
        for n, m in models.items():
            ics = per_model_oos.get(n, [])
            mean_ic = np.mean(ics) if ics else 0.01
            std_ic = np.std(ics) if len(ics) > 1 else 1.0
            icir = mean_ic / std_ic if std_ic > 0 else mean_ic
            ens[n] = {"model": m, "ic": max(icir, 0.01)}
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
            "n_stocks": n_unique_tickers, "n_obs": len(full_df),
            "pearson_correlation":round(corr,4),"spearman_rank_corr":round(rc,4),
            "mean_ic":round(ic_per.mean(),4),"ic_std":round(ic_per.std(),4),
            "ic_ir":round(ic_per.mean()/ic_per.std(),4) if ic_per.std()>0 else 0,
            "mean_ls_return":round(np.mean(ls_ret),4) if ls_ret else 0,
            "hit_rate":round((ic_per>0).mean(),4),
            "ls_returns_series":[round(r,4) for r in ls_ret],
            "ic_series":[round(float(v),4) for v in ic_per.values],
            "per_model_ic":{n:round(np.mean(v),4) for n,v in per_model_oos.items() if v}}
    else:
        oos_metrics = {"n_predictions":0,"n_stocks":0,"n_obs":0,"per_model_ic":{},"mean_ic":0,"ic_std":0,"ic_ir":0,"hit_rate":0,"spearman_rank_corr":0,"mean_ls_return":0}
    mean_ls = oos_metrics.get("mean_ls_return", 0)
    if mean_ls is None or float(mean_ls) == 0:
        n_pred = oos_metrics.get("n_predictions", 0)
        n_per = oos_metrics.get("n_periods", 0)
        if n_pred == 0:
            logger.warning("walk_forward_train: mean_ls_return=0 because no OOS predictions (oos_df empty)")
        else:
            logger.warning(
                "walk_forward_train: mean_ls_return=0 with n_predictions=%s, n_periods=%s -> need each period to have >=10 stocks to compute long-short; predicted_return_pct will be 0",
                n_pred, n_per,
            )
    if callback:
        callback(f"Ensemble OOS Rank IC: {oos_metrics.get('spearman_rank_corr','?')}")
        for n,ic in oos_metrics.get("per_model_ic",{}).items(): callback(f"  {n}: IC={ic:.4f}")

    # Final ensemble on ALL data: fit pipeline on X_all once, reuse same transforms at inference.
    if callback: callback("Training final ensemble...")
    X_all = full_df[feat_cols].copy(); y_all = full_df["forward_return"].copy()
    med_final = X_all.median()
    X_all = X_all.fillna(med_final).replace([np.inf,-np.inf],np.nan).fillna(med_final)
    fitted_winsorizer = None
    if winsorize:
        fitted_winsorizer = Winsorizer(quantile=winsorize_q)
        fitted_winsorizer.fit(X_all, feat_cols)
        X_all = fitted_winsorizer.transform(X_all)
    fitted_rank = RankTransformer()
    fitted_rank.fit(X_all, feat_cols)
    X_all = fitted_rank.transform(X_all)
    fitted_sector_neutralizer = None
    if config.get("sector_neutralization", True) and "sector" in full_df.columns:
        X_all = X_all.copy()
        X_all["sector"] = full_df["sector"].values
        fitted_sector_neutralizer = SectorNeutralizer()
        fitted_sector_neutralizer.fit(X_all, feat_cols, "sector")
        X_all = fitted_sector_neutralizer.transform(X_all, "sector").drop(columns=["sector"], errors="ignore")
    fitted_decorrelation = None
    if config.get("feature_decorrelation"):
        method = config.get("decorrelation_method", "pca")
        var_ratio = float(config.get("pca_variance_ratio", 0.95))
        dec = FeatureDecorrelator(method=method, variance_ratio=var_ratio)
        dec.fit(X_all, feat_cols)
        fitted_decorrelation = dec.fitted_
        X_all = dec.transform(X_all)
    final_models = {}
    for name,m in _get_models(config).items():
        try:
            m.fit(X_all,y_all)
            ics = per_model_oos.get(name, [])
            mean_ic = np.mean(ics) if ics else 0.01
            std_ic = np.std(ics) if len(ics) > 1 else 1.0
            icir = mean_ic / (std_ic + 1e-6) if std_ic >= 0 else mean_ic
            final_models[name] = {"model": m,
                "ic": max(icir, 0.01),
                "cv_r2": np.mean(ics) or 0}
            if callback: callback(f"  {name}: trained, OOS IC={final_models[name]['ic']:.4f}")
        except Exception as e:
            logger.warning("walk_forward final fit: %s failed: %s", name, e)
    feat_imp = _get_feature_importance(final_models,feat_cols)

    # Persist run metrics + per-period cross-section sizes into DB
    try:
        from core import portfolio as _pf_run
        _pf_run.save_model_run_metrics(
            {**oos_metrics, "n_features": len(feat_cols)},
            cross_section_sizes,
            horizon_months=H,
        )
    except Exception as e:
        logger.debug("walk_forward_train: save_model_run_metrics failed: %s", e)

    return (final_models, med_final, feat_cols, feat_imp, oos_metrics,
            fitted_decorrelation, fitted_winsorizer, fitted_rank, fitted_sector_neutralizer,
            cross_section_sizes)

# ══════════════════════════════════════════════════════════════
# CURRENT PREDICTIONS → ALPHA SCORE + RANK
# ══════════════════════════════════════════════════════════════
def predict_current(models_dict, medians, feat_cols, prices, fundamentals_db,
                    macro, sector_map, tickers, yf_info=None, callback=None, config=None, as_of_date=None,
                    macro_by_region=None, fitted_decorrelation=None, alpha_spread=None,
                    fitted_winsorizer=None, fitted_rank_transformer=None, fitted_sector_neutralizer=None):
    """Generate current alpha scores and ranks using the trained ensemble. Model output is an alpha score (ranking signal), not a direct return.
    Fitted transformers (winsorizer, rank, sector_neutralizer, decorrelation) must be applied in order; no refitting at inference."""
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
    try:
        from .data import get_region_for_ticker as _get_region_pred
    except Exception:
        _get_region_pred = None
    df = build_features_asof(
        prices, fundamentals_db, macro, ref_date, tickers,
        macro_by_region=macro_by_region, get_region=_get_region_pred if macro_by_region else None,
    )
    df = add_sector_interactions(df,sector_map)
    if yf_info:
        df["sector"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("sector","Unknown"))
        df["name"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("shortName",t))
        df["current_price"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("currentPrice"))
        df["target_mean"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("targetMeanPrice"))
        df["recommendation"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("recommendationKey"))
        df["num_analysts"] = df["ticker"].map(lambda t: yf_info.get(t,{}).get("numberOfAnalystOpinions"))

    X = df[feat_cols].fillna(medians).replace([np.inf,-np.inf],np.nan).fillna(medians)
    # Apply fitted transformers only (no refitting) in same order as training.
    if fitted_winsorizer is not None:
        X = fitted_winsorizer.transform(X)
    elif winsorize:
        X = _apply_winsorization(X, feat_cols, winsorize_q)
    if fitted_rank_transformer is not None:
        X = fitted_rank_transformer.transform(X)
    else:
        X = rank_features(X, feat_cols)
    if config.get("sector_neutralization", True) and "sector" in df.columns:
        if fitted_sector_neutralizer is not None:
            X = X.copy()
            X["sector"] = df["sector"].values
            X = fitted_sector_neutralizer.transform(X, "sector").drop(columns=["sector"], errors="ignore")
        else:
            X_neut = X.copy()
            X_neut["sector"] = df["sector"].values
            X = sector_neutralize(X_neut, feat_cols, "sector").drop(columns=["sector"], errors="ignore")
    if config.get("feature_decorrelation") and fitted_decorrelation is not None:
        X, _ = decorrelate_features(X, feat_cols, fitted_transformer=fitted_decorrelation)
    elif config.get("feature_decorrelation") and fitted_decorrelation is None:
        logger.warning("predict_current: feature_decorrelation enabled but no fitted_transformer provided; skipping decorrelation")
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

    # Alpha score: model output is a ranking signal, not a direct return forecast.
    df["alpha_score_raw"] = preds
    df["alpha_score_raw"] = df["alpha_score_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Ranking signal: derived from raw score after factor neutralization.
    df["alpha_score"] = factor_neutralize_scores(
        df,
        score_col="alpha_score_raw",
        sector_col="sector"
    )
    df["alpha_score"] = df["alpha_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.sort_values("alpha_score", ascending=False).reset_index(drop=True)
    df["alpha_rank"] = range(1, len(df) + 1)
    df["rank"] = df["alpha_rank"]  # backward compat

    # Expected return estimate: scale alpha score by historical long-short spread (not raw model output).
    # Use a minimum spread so weak models still produce interpretable expected returns in the UI.
    minimum_alpha_spread = 0.05  # conservative 5% long-short spread typical in equity factor models
    if alpha_spread is None:
        alpha_spread = minimum_alpha_spread
    alpha_spread = max(float(alpha_spread), minimum_alpha_spread)
    pred_arr = np.asarray(preds)
    z_score = (pred_arr - np.mean(pred_arr)) / (np.std(pred_arr) + 1e-9)
    expected_ret_pct = 100.0 * z_score * (alpha_spread / 2.0)
    df["expected_return_estimate_pct"] = np.round(expected_ret_pct, 2)
    df["predicted_return_pct"] = df["expected_return_estimate_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Confidence = z-score of raw alpha (relative conviction)
    med, std = np.median(preds), np.std(preds)
    df["confidence"] = ((preds - med) / std).round(2) if std > 0 else 0
    df["confidence"] = df["confidence"].fillna(0.0)
    # Reliability: combine alpha, confidence, and model agreement (prioritize high agreement)
    ar, cr = df["alpha_score"], df["confidence"]
    a_norm = (ar - ar.min()) / (ar.max() - ar.min() + 1e-9)
    c_norm = (cr - cr.min()) / (cr.max() - cr.min() + 1e-9)
    df["reliability_score"] = (0.4 * a_norm + 0.3 * c_norm + 0.3 * df["model_agreement_score"]).round(3)
    df["reliability_score"] = df["reliability_score"].fillna(0.0)
    df["predicted_return_pct"] = df["predicted_return_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["expected_return_estimate_pct"] = df.get("expected_return_estimate_pct", df["predicted_return_pct"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

            # RandomForest: use feature_importances × signed deviation as proxy (approximate)
            elif hasattr(m, "feature_importances_"):
                try:
                    imp = m.feature_importances_
                    vals = X_ranked.iloc[0].values
                    contribs = imp * (vals - 0.5) * 2  # Scale to meaningful range
                    model_exp["approximation"] = True
                    model_exp["approximation_note"] = (
                        "RF contributions are approximate (importance × deviation). "
                        "Use LightGBM/XGBoost SHAP values for reliable explanations."
                    )
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
# Pipeline: expected return estimate (from alpha score × historical long-short) → return projection → price projection → UI.
# predicted_return_pct is the expected_return_estimate_pct: scaled from alpha score using mean_ls_return (not raw model output).
# So predicted_return_pct=12 means ~12% expected return over H months (estimate), NOT annualized.
# We compound explicitly: projected_price = current_price * (1+r)^(m/H).
# When all_horizon_results is provided, each display horizon uses that horizon's prediction when available.
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
def train_simple(prices, yf_fundamentals, macro, callback=None, sentiment_scores=None, macro_by_region=None):
    """Simple mode: technical + sentiment features, walk-forward with FORWARD returns (no FMP)."""
    try:
        from core.engine_config import get_model_settings
        cfg = get_model_settings()
        if cfg and cfg.get("deterministic_mode", False):
            set_global_seed(cfg.get("global_seed", GLOBAL_SEED))
    except Exception as e:
        logger.warning("train_simple: get_model_settings failed: %s", e)

    if callback:
        callback("Simple ensemble: technical + sentiment (forward targets)...")
    logger.info("train_simple: start (full universe=%d tickers)", len(yf_fundamentals))
    try:
        from .data import get_region_for_ticker
    except Exception:
        get_region_for_ticker = None
    def _region_macro(ticker):
        if macro_by_region and get_region_for_ticker:
            region = get_region_for_ticker(ticker)
            return macro_by_region.get(region) or macro_by_region.get("US") or macro
        return macro

    if prices is None or not hasattr(prices, "index") or len(prices.index) == 0:
        logger.warning("train_simple: no price data available")
        return None, None, None, None, None, None

    horizon_months = 12
    horizon_days = horizon_months * 21  # ~252 trading sessions

    # 1) Tickers avec au moins 2 ans de prix
    valid_tickers = []
    for ticker in yf_fundamentals:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices[(ticker, "Close")].dropna()
            else:
                continue
            if len(close) >= 504:
                valid_tickers.append(ticker)
        except Exception as e:
            logger.debug("train_simple: price check failed for %s: %s", ticker, e)
    if callback:
        callback(f"  {len(valid_tickers)} tickers with ≥2y of prices")

    all_dates = prices.index.tolist()
    if len(all_dates) <= horizon_days:
        logger.warning("train_simple: not enough history for forward horizon")
        return None, None, None, None, None, None

    # Dates de rebalancement (tous les ~3 mois, en s'arrêtant horizon_days avant la fin)
    rebal_dates = []
    for i in range(0, len(all_dates) - horizon_days, 63):
        rebal_dates.append(all_dates[i])
    if len(rebal_dates) < 4:
        if callback:
            callback("Not enough rebalancing dates for walk-forward")
        return None, None, None, None, None, None

    # 2) Construire les features et FORWARD returns sur plusieurs périodes
    all_records = []
    for rd_idx, rd in enumerate(rebal_dates):
        if callback and rd_idx % 4 == 0:
            callback(f"  Building features: period {rd_idx+1}/{len(rebal_dates)}")
        for ticker in valid_tickers:
            try:
                if not isinstance(prices.columns, pd.MultiIndex):
                    continue
                close = prices[(ticker, "Close")].dropna()
                # Jusqu'à la date rd
                close_to_rd = close[close.index <= rd]
                if len(close_to_rd) < 252:
                    continue

                fund = yf_fundamentals.get(ticker, {})
                row = {
                    "ticker": ticker,
                    "name": fund.get("shortName", ticker),
                    "sector": fund.get("sector", "Unknown"),
                    "period_idx": rd_idx,
                }

                c = close_to_rd
                row["momentum_12_1"] = c.iloc[-21] / c.iloc[-252] - 1
                row["return_1m"] = c.pct_change(21).iloc[-1]
                row["return_3m"] = c.pct_change(63).iloc[-1]
                row["return_6m"] = c.pct_change(126).iloc[-1]
                daily = c.pct_change().dropna()
                row["volatility_3m"] = daily.tail(63).std() * np.sqrt(252)
                row["volatility_12m"] = daily.tail(252).std() * np.sqrt(252)
                row["price_vs_ma50"] = c.iloc[-1] / c.tail(50).mean() - 1
                row["price_vs_ma200"] = c.iloc[-1] / c.tail(200).mean() - 1
                row["drawdown_from_high"] = c.iloc[-1] / c.tail(252).max() - 1

                if sentiment_scores:
                    row["news_sentiment"] = sentiment_scores.get(ticker, 0.0)

                sec = fund.get("sector", "")
                rm = _region_macro(ticker)
                row["macro_fed"] = rm.get("policy_rate") or rm.get("fed_funds_rate", 4)
                row["macro_oil"] = rm.get("oil_price", 80)
                row["macro_vix"] = rm.get("vix", 22)
                row["tech_x_rates"] = (1 if sec in ["Technology", "Communication Services"] else 0) * row["macro_fed"]
                row["energy_x_oil"] = (1 if sec == "Energy" else 0) * row["macro_oil"]
                if sentiment_scores:
                    s = sentiment_scores.get(ticker, 0.0)
                    row["sentiment_x_momentum"] = s * row.get("momentum_12_1", 0)

                # Target: FORWARD return à partir de rd
                close_after_rd = close[close.index > rd]
                if len(close_after_rd) < horizon_days:
                    continue
                future_price = close_after_rd.iloc[min(horizon_days - 1, len(close_after_rd) - 1)]
                current_price = c.iloc[-1]
                if current_price <= 0:
                    continue
                row["forward_return"] = future_price / current_price - 1
                all_records.append(row)
            except Exception as e:
                logger.debug("train_simple: %s at period %d failed: %s", ticker, rd_idx, e)

    df = pd.DataFrame(all_records)
    if len(df) < 50:
        logger.warning("train_simple: abort (only %d rows, need ≥50)", len(df))
        return None, None, None, None, None, None

    if callback:
        callback(f"  {len(df)} observations across {df['period_idx'].nunique()} periods")

    meta = ["ticker", "name", "sector", "forward_return", "period_idx"]
    fcols = [c for c in df.columns if c not in meta and df[c].dtype in [np.float64, np.int64, float, int]]

    period_indices = sorted(df["period_idx"].unique())
    min_train = max(4, len(period_indices) // 3)

    oos_preds = []
    per_model_oos = {}

    # 3) Walk-forward OOS
    for i, tp in enumerate(period_indices):
        if i < min_train:
            continue
        trn = df[df["period_idx"].isin(period_indices[:i])]
        tst = df[df["period_idx"] == tp]
        if len(trn) < 30 or len(tst) < 5:
            continue

        Xtr = trn[fcols].copy()
        ytr = trn["forward_return"].copy()
        Xte = tst[fcols].copy()
        yte = tst["forward_return"].copy()

        med = Xtr.median()
        Xtr = Xtr.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(med)
        Xte = Xte.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(med)
        # Fit rank on train only; transform train and test (no test refitting).
        rank_t = RankTransformer()
        rank_t.fit(Xtr, fcols)
        Xtr = rank_t.transform(Xtr)
        Xte = rank_t.transform(Xte)

        models = _get_models()
        for name, m in models.items():
            try:
                m.fit(Xtr, ytr)
                p = m.predict(Xte)
                ic = stats.spearmanr(p, yte.values)[0] if len(yte) > 5 else 0
                per_model_oos.setdefault(name, []).append(ic)
            except Exception as e:
                logger.debug("train_simple OOS: %s failed: %s", name, e)
                per_model_oos.setdefault(name, []).append(0)

        ens = {n: {"model": m, "ic": np.mean(per_model_oos.get(n, [])) or 0.01} for n, m in models.items()}
        ep, _ = predict_ensemble(ens, Xte)
        for j, (_, row) in enumerate(tst.iterrows()):
            oos_preds.append({"predicted": ep[j], "actual": row["forward_return"]})

    oos_df = pd.DataFrame(oos_preds)
    if len(oos_df) > 10:
        rc, _ = stats.spearmanr(oos_df["predicted"], oos_df["actual"])
        oos_ic = round(float(rc), 4) if rc == rc else 0.0
    else:
        oos_ic = 0.0
    # Long-short spread (top quintile - bottom quintile) for return scaling
    ls_ret_simple = []
    if len(oos_df) >= 20:
        g = oos_df.sort_values("predicted", ascending=False)
        n = max(len(g) // 5, 2)
        ls_ret_simple.append(g.head(n)["actual"].mean() - g.tail(n)["actual"].mean())
    mean_ls_return = round(float(np.mean(ls_ret_simple)), 4) if ls_ret_simple else 0.0

    # 4) Entraîner l'ensemble final sur TOUTES les périodes
    if callback:
        callback("Training final ensemble (all periods)...")
    X_all = df[fcols].copy()
    y_all = df["forward_return"].copy()
    medians = X_all.median()
    X_all = X_all.fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
    fitted_rank_simple = RankTransformer()
    fitted_rank_simple.fit(X_all, fcols)
    X_all = fitted_rank_simple.transform(X_all)
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ensemble = {}
    for name, m in _get_models().items():
        try:
            m.fit(X_all, y_all)
            ic = np.mean(per_model_oos.get(name, [])) if per_model_oos.get(name) else 0.0
            ensemble[name] = {"model": m, "cv_r2": round(ic, 4), "ic": round(ic, 4)}
            if callback:
                callback(f"  {name}: OOS IC={ic:.4f}")
        except Exception as e:
            logger.warning("train_simple final fit: %s failed: %s", name, e)

    # 5) Features et prédictions à la date courante
    current_records = []
    for ticker in valid_tickers:
        try:
            if not isinstance(prices.columns, pd.MultiIndex):
                continue
            close = prices[(ticker, "Close")].dropna()
            if len(close) < 252:
                continue
            fund = yf_fundamentals.get(ticker, {})
            row = {
                "ticker": ticker,
                "name": fund.get("shortName", ticker),
                "sector": fund.get("sector", "Unknown"),
            }
            c = close
            row["momentum_12_1"] = c.iloc[-21] / c.iloc[-252] - 1
            row["return_1m"] = c.pct_change(21).iloc[-1]
            row["return_3m"] = c.pct_change(63).iloc[-1]
            row["return_6m"] = c.pct_change(126).iloc[-1]
            daily = c.pct_change().dropna()
            row["volatility_3m"] = daily.tail(63).std() * np.sqrt(252)
            row["volatility_12m"] = daily.tail(252).std() * np.sqrt(252)
            row["price_vs_ma50"] = c.iloc[-1] / c.tail(50).mean() - 1
            row["price_vs_ma200"] = c.iloc[-1] / c.tail(200).mean() - 1
            row["drawdown_from_high"] = c.iloc[-1] / c.tail(252).max() - 1
            if sentiment_scores:
                row["news_sentiment"] = sentiment_scores.get(ticker, 0.0)
            sec = fund.get("sector", "")
            rm = _region_macro(ticker)
            row["macro_fed"] = rm.get("policy_rate") or rm.get("fed_funds_rate", 4)
            row["macro_oil"] = rm.get("oil_price", 80)
            row["macro_vix"] = rm.get("vix", 22)
            row["tech_x_rates"] = (1 if sec in ["Technology", "Communication Services"] else 0) * row["macro_fed"]
            row["energy_x_oil"] = (1 if sec == "Energy" else 0) * row["macro_oil"]
            if sentiment_scores:
                s = sentiment_scores.get(ticker, 0.0)
                row["sentiment_x_momentum"] = s * row.get("momentum_12_1", 0)
            current_records.append(row)
        except Exception as e:
            logger.debug("train_simple current: %s failed: %s", ticker, e)

    results = pd.DataFrame(current_records)
    if len(results) == 0:
        logger.warning("train_simple: no current records built")
        return None, None, None, None, None, None

    Xc = results[fcols].fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
    Xc = fitted_rank_simple.transform(Xc)
    Xc = Xc.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    preds, blend = predict_ensemble(ensemble, Xc, return_per_model=True)

    n_models = sum(1 for info in ensemble.values() if info.get("model") is not None)
    if n_models > 1 and "per_model_preds" in blend:
        per_model = blend["per_model_preds"]
        pos_count = np.zeros(len(Xc))
        for p in per_model.values():
            pos_count += (np.asarray(p) > 0).astype(float)
        results["model_agreement_score"] = (pos_count / n_models).round(3)
    else:
        results["model_agreement_score"] = 1.0

    results["alpha_score_raw"] = preds
    results["alpha_score_raw"] = results["alpha_score_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    results["alpha_score"] = factor_neutralize_scores(
        results,
        score_col="alpha_score_raw",
        sector_col="sector",
    )
    results["alpha_score"] = results["alpha_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    results = results.sort_values("alpha_score", ascending=False).reset_index(drop=True)
    results["alpha_rank"] = range(1, len(results) + 1)
    results["rank"] = results["alpha_rank"]
    # Expected return estimate from historical long-short spread (after sort; use sorted alpha_score_raw)
    araw = results["alpha_score_raw"].values.astype(float)
    z_score = (araw - np.nanmean(araw)) / (np.nanstd(araw) + 1e-9)
    if mean_ls_return and float(mean_ls_return) > 0:
        results["expected_return_estimate_pct"] = np.round(100.0 * z_score * (float(mean_ls_return) / 2.0), 2)
        results["predicted_return_pct"] = results["expected_return_estimate_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        results["expected_return_estimate_pct"] = 0.0
        results["predicted_return_pct"] = 0.0

    med_pred, std_pred = np.median(preds), np.std(preds)
    results["confidence"] = ((preds - med_pred) / std_pred).round(2) if std_pred > 0 else 0
    results["confidence"] = results["confidence"].fillna(0.0)
    ar, cr = results["alpha_score"], results["confidence"]
    a_norm = (ar - ar.min()) / (ar.max() - ar.min() + 1e-9)
    c_norm = (cr - cr.min()) / (cr.max() - cr.min() + 1e-9)
    results["reliability_score"] = (
        0.4 * a_norm + 0.3 * c_norm + 0.3 * results["model_agreement_score"]
    ).round(3)
    results["reliability_score"] = results["reliability_score"].fillna(0.0)

    nan_count = results["predicted_return_pct"].isna().sum()
    if nan_count > 0:
        logger.warning("train_simple: %d NaN in predicted_return_pct, filling with 0", nan_count)
        results["predicted_return_pct"] = results["predicted_return_pct"].fillna(0.0)

    feat_imp = _get_feature_importance(ensemble, fcols)
    pm = {n: {"ic": info.get("ic", 0)} for n, info in ensemble.items()}
    per_model_ic = {n: round(info.get("ic", 0), 4) for n, info in ensemble.items()}
    hit_rate = (
        round((oos_df["predicted"] * oos_df["actual"] > 0).mean(), 4) if len(oos_df) > 0 else 0.5
    )
    oos_metrics = {
        "mode": "simple_walk_forward",
        "n_stocks": len(results),
        "n_features": len(fcols),
        "n_oos_predictions": len(oos_df),
        "spearman_rank_corr": oos_ic,
        "mean_ic": oos_ic,
        "ic_ir": 0,
        "hit_rate": hit_rate,
        "mean_ls_return": mean_ls_return,
        "per_model": pm,
        "per_model_ic": per_model_ic,
        "blend": blend,
    }
    if callback:
        callback(f"Simple walk-forward done: {len(results)} stocks, OOS IC={oos_ic:.4f}")
    return ensemble, medians, fcols, results, feat_imp, oos_metrics

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
    if oos_metrics.get("is_degraded") or mode in ("simple_ensemble", "simple_walk_forward"):
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
    from .data import fetch_all_data, load_macro_history_from_db
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
    min_wf_years = max(int(config.get("training_window_years", 3)), 3)
    max_horizon_years = max(horizons_cfg) // 12 if horizons_cfg else 5
    data_years = min(max_horizon_years + min_wf_years + 1, max_history)
    if callback: callback(f"Loading up to {data_years}y data (max {max_history}y) for horizons {horizons_cfg}...")
    alldata = fetch_all_data(years=data_years, callback=callback)
    tickers = alldata["tickers"]; prices = alldata["prices"]
    yf_fund = alldata["fundamentals"]
    macro_by_region = alldata.get("macro_by_region") or {}
    sentiment = alldata.get("sentiment",{})
    # Macro: single source of truth = existing DB (mirror fundamentals behavior)
    macro_df = load_macro_history_from_db()
    if macro_df is not None and len(macro_df) > 0:
        macro_latest = macro_df.iloc[-1].to_dict()
    else:
        macro_latest = alldata.get("macro") or {}
    macro = macro_latest  # for backward compat in _run_simple / detect_market_regime / cache
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
        try:
            from core.engine_config import get_model_settings, init_default_config
            init_default_config()
            config = get_model_settings()
            if config and config.get("deterministic_mode", False):
                set_global_seed(config.get("global_seed", GLOBAL_SEED))
        except Exception:
            config = {}
        config = config or {}
        fmp_works = False
        try:
            test = fetch_fmp_quarterly("AAPL", fmp_key)
            fmp_works = len(test) > 0
        except Exception as e:
            logger.debug("run_full_pipeline: FMP test (AAPL) failed: %s", e)
        used_yahoo_fallback = False
        if fmp_works:
            if callback:
                callback("Full mode: FMP + walk-forward ensemble")
            fund_db = fetch_all_fundamentals(list(yf_fund.keys()), fmp_key, callback)
        else:
            if callback:
                callback("FMP API unavailable (403). Building fundamentals from Yahoo Finance...")
            fund_db = build_fundamentals_from_yfinance(prices, yf_fund, callback)
            used_yahoo_fallback = True
        if len(fund_db) < 30:
            return _run_simple(prices, yf_fund, macro, sector_map, callback, sentiment, alldata.get("data_freshness"), simple_reason="too_few_fundamentals", macro_by_region=macro_by_region)
        horizons = config.get("horizons", [3, 6, 12, 24, 120])
        primary_H = config.get("primary_horizon", 12)
        ref_year = as_of_date.year if as_of_date else datetime.now().year
        all_horizon_results = {}
        total_h = len(horizons)
        for idx, H in enumerate(horizons):
            lbl = "10Y" if H == 120 else f"{H}M"
            if callback:
                callback(f"Training {lbl} horizon ({idx+1}/{total_h})...", (idx + 0.1) / total_h)
            config_h = {**config, "prediction_horizon_months": H}
            # Ensure enough history BEFORE cutoff for walk-forward folds (need min_wf_years of quarterly rebal dates)
            start_year_H = ref_year - max(H // 12, 1) - min_wf_years
            wf = walk_forward_train(prices,fund_db,macro_df,sector_map,list(fund_db.keys()),
                                   start_year=start_year_H,horizon_months=H,callback=callback,config=config_h,as_of_date=as_of_date,
                                   macro_by_region=macro_by_region)
            if wf[0] is None:
                if callback: callback(f"  {lbl}: insufficient data, skipping", (idx + 1) / total_h)
                continue
            (final_models, medians, feat_cols, feat_imp, oos_metrics,
             fitted_decorrelation, fitted_winsorizer, fitted_rank, fitted_sector_neutralizer,
             _cs_sizes) = wf
            if config_h.get("execution_mode") == "single":
                single_id = config_h.get("single_model_id")
                if single_id and single_id in final_models:
                    final_models = {single_id: final_models[single_id]}
            alpha_spread = oos_metrics.get("mean_ls_return")
            if alpha_spread is None or float(alpha_spread) <= 0:
                logger.debug("predict_current: mean_ls_return=%s -> will use minimum_alpha_spread (0.05) for expected return scaling", alpha_spread)
            else:
                logger.info("predict_current: alpha_spread=%.4f (from oos mean_ls_return) -> scaling alpha score to expected return pct", float(alpha_spread))
            results_h, blend = predict_current(
                final_models, medians, feat_cols, prices, fund_db, macro_df, sector_map, list(yf_fund.keys()),
                yf_info=yf_fund, callback=callback, config=config_h, as_of_date=as_of_date,
                macro_by_region=macro_by_region, fitted_decorrelation=fitted_decorrelation, alpha_spread=alpha_spread,
                fitted_winsorizer=fitted_winsorizer, fitted_rank_transformer=fitted_rank,
                fitted_sector_neutralizer=fitted_sector_neutralizer,
            )
            if sentiment: results_h["news_sentiment"] = results_h["ticker"].map(sentiment).fillna(0)
            all_horizon_results[H] = {"results": results_h,"feat_imp": feat_imp,"oos_metrics": oos_metrics,"blend": blend}
            if callback: callback(f"  {lbl} done: IC={oos_metrics.get('spearman_rank_corr','?')} | {len(results_h)} stocks", (idx + 1) / total_h)
        if not all_horizon_results:
            return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment,alldata.get("data_freshness"), simple_reason="all_horizons_failed", macro_by_region=macro_by_region)
        # Choose primary horizon preferring one with real OOS predictions.
        primary = all_horizon_results.get(primary_H)
        if primary is None or primary.get("oos_metrics", {}).get("n_predictions", 0) == 0:
            # Fallback: best horizon with n_predictions>0, otherwise first.
            with_oos = [
                (H, d) for H, d in all_horizon_results.items()
                if d.get("oos_metrics", {}).get("n_predictions", 0) > 0
            ]
            if with_oos:
                # Prefer horizon closest to configured primary_H
                primary_H, primary = sorted(
                    with_oos, key=lambda x: abs(x[0] - primary_H)
                )[0]
            else:
                primary_H, primary = next(iter(all_horizon_results.items()))
        results = primary["results"]
        feat_imp = primary["feat_imp"]
        oos_metrics = primary["oos_metrics"]
        n_f = len(primary["feat_imp"]) if isinstance(primary["feat_imp"], (list, dict)) else 0
        model_info = {"mode":"walk_forward_ensemble","n_features":n_f,"n_stocks":oos_metrics.get("n_stocks",0),"blend":primary["blend"],
                      "prediction_horizon_months": primary_H,"horizons_trained": list(all_horizon_results.keys()),"primary_horizon": primary_H,
                      "per_horizon_metrics": {H: d["oos_metrics"] for H,d in all_horizon_results.items()}, **oos_metrics}
        if used_yahoo_fallback:
            model_info["fund_source"] = "yfinance"
        try:
            from core.engine_config import get_risk_settings
            regime_config = get_risk_settings()
        except Exception:
            regime_config = {}
        model_info["market_regime"] = detect_market_regime(prices, macro, regime_config=regime_config)
        _store_model_state(None, None, None, prices, fund_db, sector_map, yf_fund)
    else:
        if callback: callback("Simple mode (no FMP key)")
        return _run_simple(prices,yf_fund,macro,sector_map,callback,sentiment,alldata.get("data_freshness"), simple_reason="no_fmp_key", macro_by_region=macro_by_region)
    data_freshness = alldata.get("data_freshness")
    if data_freshness:
        model_info["data_freshness"] = data_freshness
    # Expose discovered_at for UI "NEW" badge (recent discoveries)
    if not results.empty and "ticker" in results.columns and alldata.get("fundamentals"):
        results["discovered_at"] = results["ticker"].map(
            lambda t: alldata["fundamentals"].get(t, {}).get("discovered_at")
        )
    # Persist ranking snapshot and model run for stability tracking
    try:
        from core import portfolio
        run_id = datetime.now().isoformat()
        portfolio.save_ranking_snapshot(results, run_id=run_id)
        portfolio.save_model_run(model_info, run_id=run_id)
    except Exception as e:
        logger.warning("run_full_pipeline: save_ranking_snapshot/save_model_run failed: %s", e)
    _save_cache(results,feat_imp,model_info,macro,all_horizons=all_horizon_results)
    return results,feat_imp,model_info,macro,all_horizon_results

def _run_simple(prices,yf_fund,macro,sector_map,callback=None,sentiment=None,data_freshness=None, simple_reason=None, macro_by_region=None):
    r = train_simple(prices,yf_fund,macro,callback,sentiment,macro_by_region=macro_by_region)
    if r[0] is None: return None,None,{"error":"Training failed"},macro,None
    ensemble,med,fc,results,feat_imp,oos = r
    oos["prediction_horizon_months"] = 12
    oos["simple_reason"] = simple_reason
    oos["is_degraded"] = True
    oos["degraded_warning"] = (
        "Ce ranking utilise UNIQUEMENT des features techniques (momentum, volatilité, moyennes mobiles). "
        "Les fondamentaux (PE, ROE, margins, growth...) ne sont PAS inclus. "
        "Les prédictions sont moins fiables qu'en mode complet."
    )
    if data_freshness: oos["data_freshness"] = data_freshness
    _store_model_state(ensemble, med, fc, prices, {}, sector_map, yf_fund)
    try:
        from core import portfolio
        run_id = datetime.now().isoformat()
        portfolio.save_ranking_snapshot(results, run_id=run_id)
        portfolio.save_model_run(oos, run_id=run_id)
    except Exception as e:
        logger.warning("_run_simple: save_ranking_snapshot/save_model_run failed: %s", e)
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
