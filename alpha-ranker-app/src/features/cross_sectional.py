"""
Cross-sectional feature transforms and factor neutralization.

This module centralizes:
- Rank transforms
- Sector-neutralization of features
- Factor-neutralization of alpha scores
- Sector × macro interaction features

Transformer objects (fit on training data, transform train and test) to prevent data leakage:
- Winsorizer, RankTransformer, FeatureDecorrelator, SectorNeutralizer
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# FIT/TRANSFORM TRANSFORMERS (no test-set refitting)
# ══════════════════════════════════════════════════════════════


class Winsorizer:
    """Fit bounds on training data; transform clips to those bounds. Prevents leakage from test quantiles."""

    def __init__(self, quantile: float = 0.02):
        self.quantile = quantile
        self.bounds_: Dict[str, tuple] = {}  # col -> (low, high)
        self.feat_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, feat_cols: Iterable[str]) -> "Winsorizer":
        self.feat_cols_ = [c for c in feat_cols if c in X.columns]
        for c in self.feat_cols_:
            s = X[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                self.bounds_[c] = (0.0, 1.0)
            else:
                low = float(s.quantile(self.quantile))
                high = float(s.quantile(1.0 - self.quantile))
                self.bounds_[c] = (low, high)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.feat_cols_:
            if c not in out.columns:
                continue
            low, high = self.bounds_.get(c, (None, None))
            if low is not None and high is not None:
                out[c] = out[c].clip(lower=low, upper=high)
        return out


class RankTransformer:
    """Fit percentile ranks on training data; transform maps new values to percentile in train distribution. Prevents leakage."""

    def __init__(self):
        self.ref_sorted_: Dict[str, np.ndarray] = {}  # col -> sorted training values
        self.feat_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, feat_cols: Iterable[str]) -> "RankTransformer":
        self.feat_cols_ = [c for c in feat_cols if c in X.columns]
        for c in self.feat_cols_:
            s = X[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                self.ref_sorted_[c] = np.array([0.0])
            else:
                self.ref_sorted_[c] = np.sort(s.values.astype(float))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.feat_cols_:
            if c not in out.columns:
                continue
            ref = self.ref_sorted_.get(c)
            if ref is None or len(ref) == 0:
                continue
            vals = np.asarray(out[c].values, dtype=float)
            nan_mask = np.isnan(vals) | np.isinf(vals)
            idx = np.searchsorted(ref, np.where(nan_mask, np.nanmedian(ref), vals), side="right")
            pct = np.clip(idx / len(ref), 0.0, 1.0)
            pct = np.where(nan_mask, 0.5, pct)
            out[c] = pct
        return out


class SectorNeutralizer:
    """Fit sector means on training data; transform subtracts those means. Prevents leakage from test sector means."""

    def __init__(self):
        self.sector_means_: Dict[str, Dict[str, float]] = {}  # col -> {sector: mean}
        self.feat_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, feat_cols: Iterable[str], sector_col: str = "sector") -> "SectorNeutralizer":
        self.feat_cols_ = [c for c in feat_cols if c in X.columns]
        if sector_col not in X.columns:
            return self
        for c in self.feat_cols_:
            means = X.groupby(sector_col)[c].mean().to_dict()
            self.sector_means_[c] = {k: float(v) for k, v in means.items()}
        return self

    def transform(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
        out = X.copy()
        if sector_col not in out.columns:
            return out
        sectors = out[sector_col].astype(str)
        for c in self.feat_cols_:
            if c not in out.columns:
                continue
            means_map = self.sector_means_.get(c, {})
            sub = sectors.map(lambda s: means_map.get(s, 0.0))
            out[c] = out[c] - sub
        return out


class FeatureDecorrelator:
    """Fit PCA (or similar) on training data; transform uses fitted projection. Prevents leakage."""

    def __init__(self, method: str = "pca", variance_ratio: float = 0.95):
        self.method = method
        self.variance_ratio = variance_ratio
        self.fitted_: Optional[object] = None
        self.feat_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, feat_cols: List[str]) -> "FeatureDecorrelator":
        self.feat_cols_ = [c for c in feat_cols if c in X.columns]
        if len(self.feat_cols_) < 2:
            return self
        out, self.fitted_ = decorrelate_features(
            X, self.feat_cols_, method=self.method, variance_ratio=self.variance_ratio
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.fitted_ is None or len(self.feat_cols_) < 2:
            return X.copy()
        out, _ = decorrelate_features(
            X, self.feat_cols_, fitted_transformer=self.fitted_
        )
        return out


def rank_features(df: pd.DataFrame, feat_cols: Iterable[str]) -> pd.DataFrame:
    """Cross-sectional rank transform: convert raw features to percentile ranks."""
    ranked = df.copy()
    for c in feat_cols:
        if c in ranked.columns and ranked[c].dtype in [np.float64, np.int64, float, int]:
            ranked[c] = ranked[c].rank(pct=True, method="average")
    return ranked


def decorrelate_features(
    X: pd.DataFrame,
    feat_cols: List[str],
    method: str = "pca",
    variance_ratio: float = 0.95,
    fitted_transformer: Optional[object] = None,
) -> tuple:
    """
    Remove redundancy via decorrelation. Returns (transformed_df, fitted_transformer).
    If fitted_transformer is provided (e.g. from training), use it to transform; otherwise fit on X.
    method: 'pca' -> PCA projection (orthogonal components).
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return X.copy(), None
    cols = [c for c in feat_cols if c in X.columns]
    if len(cols) < 2:
        return X.copy(), None
    X_sub = X[cols].copy().fillna(X[cols].median())
    X_sub = X_sub.replace([np.inf, -np.inf], np.nan).fillna(X_sub.median())
    n_components = min(len(cols), max(1, int(X_sub.shape[0] * 0.5)))
    if fitted_transformer is not None:
        X_dec = fitted_transformer.transform(X_sub)
        out = X.copy()
        for j, c in enumerate(cols):
            if j < X_dec.shape[1]:
                out[c] = X_dec[:, j]
        return out, fitted_transformer
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    if method == "pca":
        n_comp = min(len(cols), X_scaled.shape[0] - 1, X_scaled.shape[1])
        n_comp = max(1, n_comp)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_scaled)
        if variance_ratio < 1.0 and pca.explained_variance_ratio_.size > 0:
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_keep = max(1, min(int(np.searchsorted(cumvar, variance_ratio) + 1), n_comp))
            pca = PCA(n_components=n_keep, random_state=42)
            pca.fit(X_scaled)
        X_dec = pca.transform(X_scaled)
        class _Fitted:
            def __init__(self, pca, scaler, cols):
                self.pca = pca
                self.scaler = scaler
                self.cols = cols
            def transform(self, X_new):
                X_s = X_new[self.cols].copy().fillna(X_new[self.cols].median()).replace([np.inf, -np.inf], np.nan).fillna(0)
                return self.pca.transform(self.scaler.transform(X_s))
        fitted = _Fitted(pca, scaler, cols)
    else:
        fitted = None
        X_dec = X_scaled
    out = X.copy()
    for j, c in enumerate(cols):
        if j < X_dec.shape[1]:
            out[c] = X_dec[:, j]
        else:
            out[c] = 0.0
    return out, fitted


def sector_neutralize(df: pd.DataFrame, feat_cols: Iterable[str], sector_col: str = "sector") -> pd.DataFrame:
    """Sector-neutralize features: subtract sector mean from each feature."""
    neutralized = df.copy()
    if sector_col not in neutralized.columns:
        return neutralized
    for c in feat_cols:
        if c in neutralized.columns and neutralized[c].dtype in [np.float64, np.int64, float, int]:
            sector_mean = neutralized.groupby(sector_col)[c].transform("mean")
            neutralized[c] = neutralized[c] - sector_mean
    return neutralized


def factor_neutralize_scores(
    df: pd.DataFrame,
    score_col: str = "alpha_score",
    factor_cols: Optional[List[str]] = None,
    sector_col: str = "sector",
    min_obs: int = 30,
) -> pd.Series:
    """
    Cross-sectional factor neutralization of alpha scores.

    Removes linear exposure of the alpha score to common risk factors
    (e.g., size, momentum, volatility, sector dummies) via a single-step
    cross-sectional regression and returns the residuals.
    """
    if score_col not in df.columns:
        return df.get(score_col, pd.Series(index=df.index))

    scores = df[score_col].astype(float)
    valid_idx = scores.replace([np.inf, -np.inf], np.nan).notna()
    if factor_cols is None:
        # Default factor set based on typical equity risk factors.
        default_candidates = [
            "log_market_cap",  # size
            "volatility_12m",  # risk
            "momentum_12_1",  # momentum
            "return_12m",  # trend/quality proxy
            "dividend_yield",  # value / income
        ]
        factor_cols = [
            c
            for c in default_candidates
            if c in df.columns and df[c].dtype in [np.float64, np.int64, float, int]
        ]

    if not factor_cols:
        # Nothing to neutralize against
        return scores

    X = df.loc[valid_idx, factor_cols].copy()
    # Add sector dummies if available
    if sector_col and sector_col in df.columns:
        sec = df.loc[valid_idx, sector_col].astype(str)
        dums = pd.get_dummies(sec, prefix="sec", drop_first=True)
        if len(dums.columns) > 0:
            X = pd.concat([X, dums], axis=1)

    # Drop columns that are all NaN or constant
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]

    if X.shape[1] == 0 or valid_idx.sum() < max(min_obs, X.shape[1] + 1):
        return scores

    # Fill remaining NaNs with column medians
    X = X.fillna(X.median())
    # Standardize factors to stabilize regression
    std = X.std(ddof=0).replace(0, 1)
    X_std = (X - X.mean()) / std

    y = scores.loc[valid_idx].values
    try:
        X_mat = np.column_stack([np.ones(len(X_std)), X_std.values])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        y_hat = X_mat @ beta
        resid = y - y_hat
        neutral_scores = scores.copy()
        neutral_scores.loc[valid_idx] = resid
        # Re-standardize residuals cross-sectionally for stability
        mu, sigma = np.median(neutral_scores), np.std(neutral_scores)
        if sigma > 0:
            neutral_scores = (neutral_scores - mu) / sigma
        return neutral_scores
    except Exception:
        # On failure, fall back to original scores
        return scores


def add_sector_interactions(df: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Add sector × macro interaction features."""
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
    it = df["sector"].isin(["Technology", "Communication Services"]).astype(int)
    ie = (df["sector"] == "Energy").astype(int)
    id_ = df["sector"].isin(["Utilities", "Consumer Staples", "Health Care"]).astype(int)
    iff = (df["sector"] == "Financials").astype(int)
    fed = df.get("macro_fed_funds_rate", pd.Series(4.0, index=df.index))
    oil = df.get("macro_oil_price", pd.Series(80, index=df.index))
    vix = df.get("macro_vix", pd.Series(20, index=df.index))
    df["tech_x_rates"] = it * fed
    df["energy_x_oil"] = ie * oil
    df["defensive_x_vix"] = id_ * vix
    df["financial_x_curve"] = iff * df.get("macro_yield_curve_slope", pd.Series(0, index=df.index))
    return df

