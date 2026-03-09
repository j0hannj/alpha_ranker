"""
Price-based features and target construction.

This module is split out from the original `core.model` to make the
feature factory composable and reusable across multiple models.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def compute_forward_return(prices, ticker: str, from_date, months: int = 12) -> Optional[float]:
    """Compute actual forward return. Target for supervised learning."""
    try:
        close = prices[(ticker, "Close")].dropna() if isinstance(prices.columns, pd.MultiIndex) else None
        if close is None:
            return None
        ft = pd.Timestamp(from_date)
        tt = ft + pd.DateOffset(months=months)
        af = close[close.index >= ft]
        at_ = close[close.index >= tt]
        if len(af) == 0 or len(at_) == 0:
            return None
        return at_.iloc[0] / af.iloc[0] - 1 if af.iloc[0] > 0 else None
    except Exception:
        return None

