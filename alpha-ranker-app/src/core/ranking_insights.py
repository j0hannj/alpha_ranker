"""
Ranking insights: delta analysis, stability index, movement classification, and AI-style commentary.

Uses ranking_history to compute:
- rank_delta, alpha_delta, confidence_delta vs previous run
- stability_index (rolling std of rank position; low = stable)
- movement_classification: very_stable | stable | moderately_changing | highly_volatile | rapidly_improving | rapidly_declining
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# Classification thresholds
STABILITY_LOW_STD = 2.0    # rank std below this → stable
STABILITY_MED_STD = 5.0    # below this → moderately changing
# rank_delta thresholds for "rapidly improving/declining"
RAPID_IMPROVE = 8         # rank_delta >= +8 → rapidly improving
RAPID_DECLINE = -8        # rank_delta <= -8 → rapidly declining


def compute_deltas(
    current_df: pd.DataFrame,
    previous_snapshot: Optional[dict],
) -> pd.DataFrame:
    """
    Add rank_delta, alpha_delta, confidence_delta to current_df.
    previous_snapshot: dict ticker -> {rank_position, alpha_score, confidence, timestamp}.
    """
    out = current_df.copy()
    out["rank_delta"] = np.nan
    out["alpha_delta"] = np.nan
    out["confidence_delta"] = np.nan
    out["prev_rank"] = np.nan
    out["last_model_update"] = None

    if not previous_snapshot:
        return out

    for idx, row in out.iterrows():
        ticker = row.get("ticker")
        if not ticker:
            continue
        prev = previous_snapshot.get(ticker)
        if not prev:
            continue
        rank_col = "rank" if "rank" in out.columns else "alpha_rank"
        curr_rank = int(row.get(rank_col, 0))
        curr_alpha = row.get("alpha_score")
        curr_conf = row.get("confidence")
        prev_rank = int(prev.get("rank_position", 0))
        prev_alpha = prev.get("alpha_score")
        prev_conf = prev.get("confidence")

        out.at[idx, "prev_rank"] = prev_rank
        out.at[idx, "rank_delta"] = curr_rank - prev_rank  # positive = improved (rank went down in position number)
        if curr_alpha is not None and curr_alpha == curr_alpha and prev_alpha is not None:
            out.at[idx, "alpha_delta"] = float(curr_alpha) - float(prev_alpha)
        if curr_conf is not None and curr_conf == curr_conf and prev_conf is not None:
            out.at[idx, "confidence_delta"] = float(curr_conf) - float(prev_conf)
        out.at[idx, "last_model_update"] = prev.get("timestamp")

    return out


def stability_index(rank_positions: list) -> float:
    """Rolling std of rank positions. Low = stable, high = volatile. Returns np.nan if len < 2."""
    if not rank_positions or len(rank_positions) < 2:
        return np.nan
    return float(np.std(rank_positions))


def movement_classification(rank_delta: Optional[float], rank_std: float) -> str:
    """
    Classify signal into: very_stable | stable | moderately_changing | highly_volatile | rapidly_improving | rapidly_declining.
    """
    if rank_delta is not None and rank_delta == rank_delta:
        if rank_delta >= RAPID_IMPROVE:
            return "rapidly improving"
        if rank_delta <= RAPID_DECLINE:
            return "rapidly declining"
    if np.isnan(rank_std) or rank_std <= STABILITY_LOW_STD:
        return "very stable" if rank_std is not np.nan and rank_std <= 1.0 else "stable"
    if rank_std <= STABILITY_MED_STD:
        return "moderately changing"
    return "highly volatile"


def add_ranking_insights(
    current_df: pd.DataFrame,
    previous_snapshot: Optional[dict],
    history_by_ticker: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Enrich current_df with rank_delta, alpha_delta, confidence_delta, stability_index, movement_classification, ranking_commentary.
    history_by_ticker: optional dict ticker -> list of {rank_position, ...} from get_ranking_history (for stability).
    """
    out = compute_deltas(current_df, previous_snapshot)

    out["stability_index"] = np.nan
    out["movement_classification"] = ""
    out["ranking_commentary"] = ""

    rank_col = "rank" if "rank" in out.columns else "alpha_rank"
    for idx, row in out.iterrows():
        ticker = row.get("ticker")
        if not ticker:
            continue
        rank_delta = row.get("rank_delta")
        if pd.isna(rank_delta):
            rank_delta = None
        else:
            rank_delta = int(rank_delta)

        # Stability from history
        rank_std = np.nan
        if history_by_ticker and ticker in history_by_ticker:
            hist = history_by_ticker[ticker]
            positions = [h.get("rank_position") for h in hist if h.get("rank_position") is not None]
            if len(positions) >= 2:
                rank_std = stability_index(positions)
        out.at[idx, "stability_index"] = rank_std

        classification = movement_classification(rank_delta, rank_std)
        out.at[idx, "movement_classification"] = classification

        # Template commentary
        name = row.get("name") or ticker
        if rank_delta is not None:
            if rank_delta > 0:
                out.at[idx, "ranking_commentary"] = (
                    f"{ticker} moved higher in the ranking (improved by {rank_delta} positions) "
                    f"due to stronger alpha and model consensus."
                )
            elif rank_delta < 0:
                out.at[idx, "ranking_commentary"] = (
                    f"{ticker} dropped in the ranking (down {abs(rank_delta)} positions) "
                    f"due to weakening signals and declining model agreement."
                )
            else:
                out.at[idx, "ranking_commentary"] = f"{ticker} rank unchanged; signal is {classification}."
        else:
            out.at[idx, "ranking_commentary"] = f"{ticker} has no prior run to compare; first ranking."

    return out
