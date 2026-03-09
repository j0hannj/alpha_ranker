"""
Feature engineering layer for the alpha research platform.

This package contains:
- Point-in-time price and fundamental features
- Macro and sentiment feature construction
- Cross-sectional transforms (ranking, sector/factor neutralization)

Existing training code in `core.model` can be gradually migrated to use
these functions directly; for now we keep backward-compatible wrappers
to avoid breaking `run_full_pipeline`.
"""

from .price_features import compute_forward_return
from .fundamental_features import get_fundamentals_asof, build_features_asof
from .cross_sectional import (
    rank_features,
    sector_neutralize,
    factor_neutralize_scores,
    add_sector_interactions,
)

__all__ = [
    "compute_forward_return",
    "get_fundamentals_asof",
    "build_features_asof",
    "rank_features",
    "sector_neutralize",
    "factor_neutralize_scores",
    "add_sector_interactions",
]

