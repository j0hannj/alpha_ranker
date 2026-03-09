"""
Cross-sectional feature transforms and factor neutralization.

This module centralizes:
- Rank transforms
- Sector-neutralization of features
- Factor-neutralization of alpha scores
- Sector × macro interaction features
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def rank_features(df: pd.DataFrame, feat_cols: Iterable[str]) -> pd.DataFrame:
    """Cross-sectional rank transform: convert raw features to percentile ranks."""
    ranked = df.copy()
    for c in feat_cols:
        if c in ranked.columns and ranked[c].dtype in [np.float64, np.int64, float, int]:
            ranked[c] = ranked[c].rank(pct=True, method="average")
    return ranked


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

