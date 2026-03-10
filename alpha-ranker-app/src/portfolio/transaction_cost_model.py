"""
Transaction cost modeling for the portfolio decision engine.

Trades are evaluated on net expected return after costs:
  net_expected_return = expected_return - transaction_cost (as fraction of notional)
Assets are rejected only when net expected return is negative or below a small
configurable minimum (e.g. 1%). This avoids over-filtering that would leave the
portfolio builder with no buy candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransactionCostParams:
    """User-configurable transaction cost assumptions."""
    broker_fee: float = 0.0      # Fixed fee per trade (e.g. EUR)
    spread_bps: float = 10.0     # Bid-ask spread in basis points (1 bps = 0.01%)
    slippage_bps: float = 5.0    # Slippage in basis points


def estimate_transaction_cost(
    price: float,
    units: float,
    notional: Optional[float] = None,
    params: Optional[TransactionCostParams] = None,
) -> float:
    """
    Total transaction cost per trade (one-way).

    transaction_cost = estimated_spread + broker_fee + slippage

    Spread and slippage are applied to notional (price * units) in bps.
    """
    if params is None:
        params = TransactionCostParams()
    notional = notional if notional is not None else price * units
    spread_cost = notional * (params.spread_bps / 10_000)
    slippage_cost = notional * (params.slippage_bps / 10_000)
    return params.broker_fee + spread_cost + slippage_cost


def is_trade_net_expected_viable(
    expected_return_frac: float,
    transaction_cost_frac: float,
    minimum_alpha_frac: float = 0.01,
) -> bool:
    """
    Viable if net expected return (after costs) exceeds a small minimum.

    net_expected_return = expected_return_frac - transaction_cost_frac
    Reject only when net_expected_return <= minimum_alpha_frac (e.g. 1% = 0.01).
    This prevents transaction costs from eliminating almost all candidates.
    """
    net = expected_return_frac - transaction_cost_frac
    return net > minimum_alpha_frac


def is_trade_economically_viable(
    expected_return: float,
    transaction_cost: float,
    min_multiple: float = 3.0,
) -> bool:
    """
    Legacy: require expected_return >= min_multiple × transaction_cost.
    Prefer is_trade_net_expected_viable for portfolio build (net return vs minimum alpha).
    """
    if transaction_cost <= 0:
        return expected_return > 0
    return expected_return >= min_multiple * transaction_cost


def expected_return_as_fraction(expected_return_pct: float) -> float:
    """Convert expected return in percentage to fraction (e.g. 10 -> 0.10)."""
    return expected_return_pct / 100.0
