"""
Confidence-based filtering of alpha model recommendations.

The alpha model outputs a confidence score per asset (e.g. z-score of alpha).
This module filters the ranked universe by a user-chosen confidence level
before portfolio construction, so that only signals above the threshold
are considered for investment.

How confidence filtering works:
- LOW:   No filter (include all predictions). Maximizes diversification.
- MEDIUM: Include only stocks where confidence >= 0 (above-median signal).
- HIGH:  Include only stocks where confidence >= 1 (stronger conviction).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]

# Numeric thresholds applied to the "confidence" column of model results.
# LOW: no minimum (use -inf so all pass); MEDIUM: 0; HIGH: 1.
CONFIDENCE_THRESHOLDS: dict[ConfidenceLevel, float] = {
    "LOW": float("-inf"),
    "MEDIUM": 0.0,
    "HIGH": 1.0,
}


class ConfidenceFilter:
    """Filters a model results DataFrame by minimum confidence."""

    def __init__(self, level: ConfidenceLevel = "MEDIUM"):
        self.level = level.upper() if isinstance(level, str) else level
        if self.level not in CONFIDENCE_THRESHOLDS:
            self.level = "LOW"
        self.threshold = CONFIDENCE_THRESHOLDS[self.level]

    def filter(self, df: pd.DataFrame, confidence_col: str = "confidence") -> pd.DataFrame:
        """
        Return rows where confidence >= threshold.

        If confidence_col is missing or has no valid values, returns df unchanged.
        """
        if df.empty or confidence_col not in df.columns:
            return df
        return df[df[confidence_col].fillna(float("-inf")) >= self.threshold].copy()
