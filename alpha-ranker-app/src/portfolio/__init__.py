"""
Portfolio decision engine: suggestions, transaction costs, sell signals, and tracking.

- Confidence filtering (LOW / MEDIUM / HIGH) and allocation with integer shares.
- Transaction cost: reject only when net expected return (expected_return - tx_cost) <= minimum_alpha.
- Sell signal engine: target reached, stop loss, horizon, confidence drop, negative alpha.
- portfolio_database: strategy_type per asset, OPEN/SOLD status, full tracking fields.
"""

from .confidence_filter import ConfidenceFilter, CONFIDENCE_THRESHOLDS
from .allocation_engine import AllocationEngine
from .portfolio_builder import PortfolioBuilder, build_suggested_portfolio
from .transaction_cost_model import (
    TransactionCostParams,
    estimate_transaction_cost,
    is_trade_economically_viable,
    is_trade_net_expected_viable,
)
from .sell_signal_engine import evaluate_sell_signals, get_all_sell_alerts, get_all_sell_signals, SellAlert, format_sell_alert
from .portfolio_database import get_open_positions, add_position, update_position, mark_sold, STRATEGY_TYPES

__all__ = [
    "ConfidenceFilter",
    "CONFIDENCE_THRESHOLDS",
    "AllocationEngine",
    "PortfolioBuilder",
    "build_suggested_portfolio",
    "TransactionCostParams",
    "estimate_transaction_cost",
    "is_trade_economically_viable",
    "is_trade_net_expected_viable",
    "evaluate_sell_signals",
    "get_all_sell_alerts",
    "get_all_sell_signals",
    "SellAlert",
    "format_sell_alert",
    "get_open_positions",
    "add_position",
    "update_position",
    "mark_sold",
    "STRATEGY_TYPES",
]
