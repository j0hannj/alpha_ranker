"""
Portfolio suggestion system for the quantitative investment application.

This package implements realistic portfolio construction that:
- Filters recommendations by model confidence (LOW / MEDIUM / HIGH).
- Allocates capital proportionally, then converts to integer shares using
  current market prices so the total cost never exceeds the budget.

Compatible with the existing alpha model pipeline (model_results DataFrame
with columns: ticker, alpha_score, confidence, current_price, etc.).
"""

from .confidence_filter import ConfidenceFilter, CONFIDENCE_THRESHOLDS
from .allocation_engine import AllocationEngine
from .portfolio_builder import PortfolioBuilder, build_suggested_portfolio

__all__ = [
    "ConfidenceFilter",
    "CONFIDENCE_THRESHOLDS",
    "AllocationEngine",
    "PortfolioBuilder",
    "build_suggested_portfolio",
]
