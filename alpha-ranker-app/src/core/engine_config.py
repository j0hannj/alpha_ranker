"""
Engine configuration: defaults and DB-backed system config.

All model parameters, strategy settings, and portfolio rules are configurable
from the application; they are stored in the system_config table and loaded
when the alpha pipeline runs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from core import portfolio as _portfolio
except Exception:
    import core.portfolio as _portfolio  # type: ignore

# Domains stored in system_config
DOMAIN_MODEL = "model_settings"
DOMAIN_PORTFOLIO = "portfolio_settings"
DOMAIN_TRANSACTION_COST = "transaction_cost_settings"
DOMAIN_FEATURE = "feature_settings"
DOMAIN_RISK = "risk_settings"

# Available models (TCN/LSTM optional; enabled only if deps present)
MODEL_IDS = ["LightGBM", "XGBoost", "RandomForest", "Ridge", "ElasticNet", "TCN", "LSTM"]
ENSEMBLE_METHODS = ["simple_average", "ic_weighted_average", "stacked_meta_model"]
EXECUTION_MODES = ["single", "multi", "all"]

# Feature groups (columns produced by build_features_asof / feature factory)
FEATURE_GROUPS = {
    "momentum": ["return_1m", "return_3m", "return_6m", "return_12m", "momentum_12_1"],
    "volatility": ["volatility_1m", "volatility_3m", "volatility_12m", "sharpe_12m"],
    "value": ["pe_ratio", "pb_ratio", "ev_ebitda", "fcf_yield", "dividend_yield", "peg_ratio"],
    "quality": ["roe", "gross_margin", "operating_margin", "net_margin", "revenue_growth_yoy", "eps_growth_yoy", "current_ratio", "debt_to_equity"],
    "trend": ["price_vs_ma50", "price_vs_ma200", "drawdown_from_high", "distance_from_low"],
    "macro": ["macro_fed_funds_rate", "macro_oil_price", "macro_vix", "macro_yield_curve_slope", "tech_x_rates", "energy_x_oil", "defensive_x_vix", "financial_x_curve"],
    "sentiment": ["news_sentiment"],
    "size": ["log_market_cap"],
}

DEFAULT_MODEL_SETTINGS = {
    "enabled_models": ["LightGBM", "XGBoost", "RandomForest", "Ridge"],
    "execution_mode": "all",
    "single_model_id": "LightGBM",
    "ensemble_method": "ic_weighted_average",
    "prediction_horizon_months": 12,
    "lookback_days": 252,
    "training_window_years": 3,
    "retraining_frequency_months": 3,
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "ridge_alpha": 10.0,
    "elastic_net_alpha": 1.0,
    "elastic_net_l1_ratio": 0.5,
    "tcn_kernel_size": 3,
    "tcn_channels": [32, 32],
    "tcn_dropout": 0.2,
    "tcn_dilation_levels": 4,
    "lstm_units": 64,
    "lstm_dropout": 0.2,
    "winsorization": True,
    "winsorize_quantile": 0.02,
    "rank_normalization": True,
    "sector_neutralization": True,
    "feature_decorrelation": False,
    "decorrelation_method": "pca",
    "pca_variance_ratio": 0.95,
    "use_gpu": True,  # When False, force CPU for TCN/LSTM/Transformer
    # Term structure: weights for combining alpha across horizons (when multi-horizon is used)
    "horizon_weight_3m": 0.1,
    "horizon_weight_6m": 0.2,
    "horizon_weight_12m": 0.5,
    "horizon_weight_24m": 0.2,
    "deterministic_mode": True,
    "global_seed": 42,
}

DEFAULT_FEATURE_SETTINGS = {
    "momentum": True,
    "volatility": True,
    "value": True,
    "quality": True,
    "trend": True,
    "macro": True,
    "sentiment": True,
    "size": True,
}

# Strategy types and default horizons (days). DONT_SELL = no sell alerts (hold through horizon/stop).
STRATEGY_TYPES = ["SHORT_TERM", "MEDIUM_TERM", "LONG_TERM", "DONT_SELL"]
# Default horizon ranges: SHORT_TERM 30-90, MEDIUM_TERM 90-365, LONG_TERM 365+
DEFAULT_HORIZON_SHORT_TERM_MIN = 30
DEFAULT_HORIZON_SHORT_TERM_MAX = 90
DEFAULT_HORIZON_MEDIUM_TERM_MIN = 90
DEFAULT_HORIZON_MEDIUM_TERM_MAX = 365
DEFAULT_HORIZON_LONG_TERM_MIN = 365

DEFAULT_PORTFOLIO_SETTINGS = {
    "confidence_threshold": "MEDIUM",
    "max_positions": 10,
    "max_position_pct": 0.15,
    "min_return_vs_cost_multiple": 3.0,
    "holding_horizon_days": 365,
    "default_stop_loss_pct": 10.0,
    # Per-strategy default horizons (days)
    "horizon_short_term_days": 60,
    "horizon_medium_term_days": 180,
    "horizon_long_term_days": 365,
    "max_holding_duration_days": 730,
    "review_frequency_days": 30,
    # Sell signal behavior per strategy: "disabled" | "passive" (stop-loss only) | "active" (all signals)
    "sell_mode_long_term": "disabled",
    "sell_mode_medium_term": "passive",
    "sell_mode_short_term": "active",
    "sell_mode_speculative": "active",
    "sell_mode_dont_sell": "disabled",
}

DEFAULT_TRANSACTION_COST_SETTINGS = {
    "broker_fee": 0.0,
    "spread_bps": 10.0,
    "slippage_bps": 5.0,
}

DEFAULT_RISK_SETTINGS = {
    "max_sector_weight": 0.35,
    "max_turnover_pct": 0.5,
    "max_position_pct": 0.15,
    "bull_return_6m": 0.05,
    "bear_return_6m": -0.05,
    "high_vol_threshold": 0.25,
    "low_vol_threshold": 0.15,
}


def _deep_merge(base: dict, override: Optional[dict]) -> dict:
    if override is None:
        return dict(base)
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_system_config(domain: str) -> Optional[dict]:
    return _portfolio.get_system_config(domain)


def set_system_config(domain: str, value: dict) -> None:
    _portfolio.set_system_config(domain, value)


def get_model_settings() -> dict:
    """Model settings merged with defaults (for alpha pipeline)."""
    raw = get_system_config(DOMAIN_MODEL)
    return _deep_merge(DEFAULT_MODEL_SETTINGS, raw)


def get_feature_settings() -> dict:
    """Feature group toggles merged with defaults."""
    raw = get_system_config(DOMAIN_FEATURE)
    return _deep_merge(DEFAULT_FEATURE_SETTINGS, raw)


def get_portfolio_settings() -> dict:
    raw = get_system_config(DOMAIN_PORTFOLIO)
    return _deep_merge(DEFAULT_PORTFOLIO_SETTINGS, raw)


def get_transaction_cost_settings() -> dict:
    raw = get_system_config(DOMAIN_TRANSACTION_COST)
    return _deep_merge(DEFAULT_TRANSACTION_COST_SETTINGS, raw)


def get_risk_settings() -> dict:
    raw = get_system_config(DOMAIN_RISK)
    return _deep_merge(DEFAULT_RISK_SETTINGS, raw)


def get_enabled_feature_columns() -> list:
    """List of feature column names that are enabled by current feature_settings."""
    feat = get_feature_settings()
    out = []
    for group, cols in FEATURE_GROUPS.items():
        if feat.get(group, True):
            out.extend(cols)
    return list(dict.fromkeys(out))


def get_strategy_horizon_days(strategy_type: str) -> int:
    """Return default holding horizon in days for a strategy type."""
    ps = get_portfolio_settings()
    m = {
        "SHORT_TERM": ps.get("horizon_short_term_days", 60),
        "MEDIUM_TERM": ps.get("horizon_medium_term_days", 180),
        "LONG_TERM": ps.get("horizon_long_term_days", 365),
    }
    return m.get(strategy_type, ps.get("holding_horizon_days", 365))


def strategy_type_from_horizon(holding_days: int) -> str:
    """Map holding horizon (days) to strategy type using configured boundaries."""
    ps = get_portfolio_settings()
    short = ps.get("horizon_short_term_days", 60)
    med = ps.get("horizon_medium_term_days", 180)
    if holding_days <= short:
        return "SHORT_TERM"
    if holding_days <= med:
        return "MEDIUM_TERM"
    return "LONG_TERM"


def init_default_config():
    """Write default config into DB if not present (so UI has something to edit)."""
    for domain, default in [
        (DOMAIN_MODEL, DEFAULT_MODEL_SETTINGS),
        (DOMAIN_FEATURE, DEFAULT_FEATURE_SETTINGS),
        (DOMAIN_PORTFOLIO, DEFAULT_PORTFOLIO_SETTINGS),
        (DOMAIN_TRANSACTION_COST, DEFAULT_TRANSACTION_COST_SETTINGS),
        (DOMAIN_RISK, DEFAULT_RISK_SETTINGS),
    ]:
        if get_system_config(domain) is None:
            set_system_config(domain, default)
