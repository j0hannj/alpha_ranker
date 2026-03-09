"""
Transaction cost modeling for the portfolio decision engine.

Why transaction costs must be modeled:
- Every trade incurs costs (broker fees, bid-ask spread, slippage). Ignoring them
  leads to over-trading and recommendations that are not economically viable.
- The model must reject buys where expected return is too small relative to
  transaction cost (e.g. expected_return >= 3 × transaction_cost) so that only
  trades with a sufficient edge are suggested.
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


def is_trade_economically_viable(
    expected_return: float,
    transaction_cost: float,
    min_multiple: float = 3.0,
) -> bool:
    """
    Require expected_return >= min_multiple × transaction_cost (as fraction of notional).

    Example: min_multiple=3 means we only recommend a trade if the expected
    return is at least 3× the one-way transaction cost, so that after round-trip
    costs the trade still has positive expected edge.
    """
    if transaction_cost <= 0:
        return expected_return > 0
    return expected_return >= min_multiple * transaction_cost


def expected_return_as_fraction(expected_return_pct: float) -> float:
    """Convert expected return in percentage to fraction (e.g. 10 -> 0.10)."""
    return expected_return_pct / 100.0
