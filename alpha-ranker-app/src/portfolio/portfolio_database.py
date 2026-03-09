"""
Portfolio database layer for the decision engine.

Provides a clear interface over the core SQLite holdings table for:
- Open and sold positions with full tracking fields (strategy_type, target_price,
  stop_loss, holding_horizon_days, transaction_cost, status).
- User-configurable strategy_type per asset (LONG_TERM, MEDIUM_TERM, SHORT_TERM).
- Consistency with core.portfolio for get_all, add, update, so the rest of the app
  continues to work while new code can use the extended schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

# Use core portfolio for persistence; this module adds schema and helpers
try:
    from core import portfolio as core_portfolio
except Exception:
    import core.portfolio as core_portfolio  # type: ignore

# Strategy types user can set per asset
STRATEGY_TYPES = ("LONG_TERM", "MEDIUM_TERM", "SHORT_TERM")
STATUS_OPEN = "OPEN"
STATUS_SOLD = "SOLD"


def get_all_holdings() -> List[dict]:
    """Return all holdings (open and sold) with extended fields."""
    return core_portfolio.get_all()


def get_open_positions() -> List[dict]:
    """Return only positions with status OPEN (default for legacy rows)."""
    all_ = core_portfolio.get_all()
    return [h for h in all_ if (h.get("status") or STATUS_OPEN) == STATUS_OPEN]


def add_position(
    ticker: str,
    name: str,
    asset_type: str,
    units: float,
    entry_price: float,
    currency: str = "EUR",
    *,
    isin: Optional[str] = None,
    sector: Optional[str] = None,
    sectors_json: Optional[str] = None,
    strategy_type: str = "LONG_TERM",
    confidence: Optional[float] = None,
    alpha_score: Optional[float] = None,
    expected_return: Optional[float] = None,
    target_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    holding_horizon_days: Optional[int] = None,
    transaction_cost: Optional[float] = None,
) -> None:
    """Add a new position with full tracking fields."""
    core_portfolio.add(
        ticker, name, asset_type, units, entry_price, currency,
        sector=sector, sectors_json=sectors_json, isin=isin,
        strategy_type=strategy_type,
        entry_date=datetime.now().strftime("%Y-%m-%d"),
        confidence=confidence,
        alpha_score=alpha_score,
        expected_return=expected_return,
        target_price=target_price,
        stop_loss=stop_loss,
        holding_horizon_days=holding_horizon_days,
        transaction_cost=transaction_cost,
    )


def update_position(
    position_id: int,
    **kwargs,
) -> None:
    """Update a position; supports strategy_type, target_price, stop_loss, status, etc."""
    core_portfolio.update(position_id, **kwargs)


def mark_sold(position_id: int) -> None:
    """Set status to SOLD for a position."""
    core_portfolio.update(position_id, status=STATUS_SOLD)


def get_setting(key: str, default=None):
    """Delegate to core portfolio settings (e.g. transaction cost params, confidence threshold)."""
    return core_portfolio.get_setting(key, default)


def set_setting(key: str, value) -> None:
    """Store a user configuration value."""
    core_portfolio.set_setting(key, value)
