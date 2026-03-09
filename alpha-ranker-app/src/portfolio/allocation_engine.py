"""
Allocation engine: proportional capital allocation with integer share constraints.

Why integer share constraints are required:
- Most brokers do not support fractional shares for equities; positions
  must be in whole units. ETFs may support fractional trading in some
  jurisdictions; here we apply floor(alloc/price) universally for
  consistency and to never exceed the available budget.

How the budget constraint is enforced:
- We allocate proportionally to the chosen weighting (equal, risk parity,
  or alpha weight), then for each asset compute units = floor(alloc/price).
- If price > remaining budget, the asset is skipped (0 units).
- We track remaining_cash and only add positions whose cost fits within it.
- Total cost of the suggested portfolio is therefore always <= budget.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


class AllocationEngine:
    """Converts a filtered, ranked universe into integer-share positions under a budget."""

    def __init__(
        self,
        budget: float,
        weight_method: str = "alpha_weight",
        max_positions: Optional[int] = None,
        allow_fractional_shares: bool = False,
    ):
        self.budget = budget
        self.weight_method = weight_method
        self.max_positions = max_positions
        self.allow_fractional = allow_fractional_shares

    def allocate(
        self,
        df: pd.DataFrame,
        price_col: str = "current_price",
        alpha_col: str = "alpha_score",
        confidence_col: str = "confidence",
    ) -> List[dict]:
        """
        Allocate capital across assets; return list of position dicts with
        ticker, price, units, invested_amount, confidence, alpha_score.

        - Ranks by alpha_score (descending).
        - Computes weights from weight_method (equal_weight, risk_parity, alpha_weight).
        - For each asset: allocation = budget * weight; units = floor(allocation / price).
        - Skips asset if price > remaining budget or price missing/invalid.
        - Stops when max_positions is reached or budget exhausted.
        """
        if df.empty or self.budget <= 0:
            return []

        # Ensure we have a price column; drop rows without valid price
        if price_col not in df.columns:
            return []
        df = df.copy()
        df["_price"] = pd.to_numeric(df[price_col], errors="coerce")
        df = df[df["_price"].notna() & (df["_price"] > 0)]
        if df.empty:
            return []

        # Rank by alpha (higher first)
        if alpha_col in df.columns:
            df = df.sort_values(alpha_col, ascending=False).reset_index(drop=True)
        n = min(len(df), self.max_positions or len(df))
        df = df.head(n)

        # Weights
        if self.weight_method == "equal_weight":
            weights = [1.0 / n] * n
        elif self.weight_method == "risk_parity":
            vol_col = "volatility_12m"
            if vol_col in df.columns:
                vol = df[vol_col].fillna(0.2).clip(lower=0.05)
                inv_vol = 1.0 / vol.values
                weights = (inv_vol / inv_vol.sum()).tolist()
            else:
                weights = [1.0 / n] * n
        else:
            # alpha_weight: weight by max(alpha, small) then normalize
            a = df[alpha_col].fillna(0).values if alpha_col in df.columns else [1.0] * n
            a = [max(float(x), 0.01) for x in a]
            total_a = sum(a)
            weights = [x / total_a for x in a]

        remaining = self.budget
        positions: List[dict] = []

        for i, (idx, row) in enumerate(df.iterrows()):
            if remaining <= 0 or (self.max_positions and len(positions) >= self.max_positions):
                break
            price = float(row["_price"])
            if price > remaining:
                continue
            alloc = self.budget * weights[i]
            if alloc > remaining:
                alloc = remaining
            if self.allow_fractional:
                units = alloc / price
                invested = units * price
            else:
                units = int(alloc / price)
                if units <= 0:
                    continue
                invested = units * price
            if invested > remaining:
                if self.allow_fractional:
                    units = remaining / price
                    invested = units * price
                else:
                    units = int(remaining / price)
                    if units <= 0:
                        continue
                    invested = units * price
            remaining -= invested
            conf_val = row.get(confidence_col)
            conf_val = float(conf_val) if pd.notna(conf_val) else None
            alpha_val = row.get(alpha_col)
            alpha_val = float(alpha_val) if pd.notna(alpha_val) else None
            pos = {
                "ticker": row["ticker"],
                "price": round(price, 2),
                "units": units if self.allow_fractional else int(units),
                "invested_amount": round(invested, 2),
                "confidence": conf_val,
                "alpha_score": alpha_val,
            }
            positions.append(pos)
        return positions
