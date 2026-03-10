"""
Ranking insights: delta analysis, stability index, movement classification, and AI-style commentary.

Uses ranking_history (DB) to compute:
- rank_delta, alpha_delta, confidence_delta vs previous run
- stability_index (rolling std of rank position; low = stable)
- movement_classification: new | stable | moderate | volatile | improving | degrading | returning | insufficient_data
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_stability_from_history(
    results_df: pd.DataFrame,
    db_path: str,
    n_runs: int = 5,
) -> dict:
    """
    For each ticker in the current run, compute stability from DB history.
    Current run is NOT yet saved — we compare with the last N runs only.

    Returns dict: ticker -> {classification, rank_delta, stability_index, alpha_delta, confidence_delta}.

    Classification:
    - "new": ticker never seen in any run
    - "returning": seen before but not in last N runs
    - "stable" / "moderate" / "volatile": from rank std over recent runs
    - "improving" / "degrading": trend from rank slope
    - "insufficient_data": not enough history
    """
    import sqlite3
    result = {}
    if results_df is None or results_df.empty:
        return result
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception:
        return {row["ticker"]: {"classification": "new", "rank_delta": None, "stability_index": None, "alpha_delta": None, "confidence_delta": None}
                for _, row in results_df.iterrows()}

    # Last N run timestamps (current run not in DB yet)
    runs = conn.execute(
        "SELECT DISTINCT timestamp FROM ranking_history ORDER BY timestamp DESC LIMIT ?",
        (n_runs,),
    ).fetchall()
    run_ts_list = [r[0] for r in runs]

    if not run_ts_list:
        conn.close()
        return {row["ticker"]: {"classification": "new", "rank_delta": None, "stability_index": None, "alpha_delta": None, "confidence_delta": None}
                for _, row in results_df.iterrows()}

    placeholders = ",".join("?" * len(run_ts_list))
    history = conn.execute(
        f"SELECT ticker, timestamp, rank_position FROM ranking_history WHERE timestamp IN ({placeholders}) ORDER BY timestamp",
        run_ts_list,
    ).fetchall()

    # Last run snapshot for alpha_delta and confidence_delta (ranking_history has alpha_score, confidence)
    last_ts = run_ts_list[0]  # most recent run in DB
    prev_row = conn.execute(
        "SELECT ticker, alpha_score, rank_position, confidence FROM ranking_history WHERE timestamp = ?",
        (last_ts,),
    ).fetchall()
    prev_by_ticker = {r[0]: {"alpha_score": r[1], "rank_position": r[2], "confidence": r[3]} for r in prev_row}

    all_known = set(
        r[0] for r in conn.execute("SELECT DISTINCT ticker FROM ranking_history").fetchall()
    )
    conn.close()

    # ticker -> list of (run_ts, rank) ordered by time
    ticker_history: dict = {}
    for row in history:
        t, ts, rank = row[0], row[1], row[2]
        ticker_history.setdefault(t, []).append((ts, int(rank) if rank is not None else None))
    for t in ticker_history:
        ticker_history[t].sort(key=lambda x: x[0])

    rank_col = "rank" if "rank" in results_df.columns else "alpha_rank"
    for _, row in results_df.iterrows():
        ticker = row.get("ticker")
        if not ticker:
            continue
        current_rank = row.get(rank_col)
        if current_rank is not None and pd.notna(current_rank):
            current_rank = int(current_rank)
        else:
            current_rank = None

        if ticker not in all_known:
            result[ticker] = {"classification": "new", "rank_delta": None, "stability_index": None, "alpha_delta": None, "confidence_delta": None}
            continue

        ranks_in_runs = ticker_history.get(ticker, [])
        if not ranks_in_runs:
            result[ticker] = {"classification": "returning", "rank_delta": None, "stability_index": None, "alpha_delta": None, "confidence_delta": None}
            continue

        last_rank = ranks_in_runs[-1][1]
        delta = (current_rank - last_rank) if current_rank is not None and last_rank is not None else None
        all_ranks = [r[1] for r in ranks_in_runs if r[1] is not None]
        if current_rank is not None:
            all_ranks.append(current_rank)

        # alpha_delta and confidence_delta: current (from results_df) vs last run (from DB)
        alpha_delta = None
        confidence_delta = None
        prev = prev_by_ticker.get(ticker)
        if prev is not None:
            curr_alpha = row.get("alpha_score")
            curr_conf = row.get("confidence")
            if curr_alpha is not None and pd.notna(curr_alpha) and prev.get("alpha_score") is not None:
                try:
                    alpha_delta = float(curr_alpha) - float(prev["alpha_score"])
                except (TypeError, ValueError):
                    pass
            if curr_conf is not None and pd.notna(curr_conf) and prev.get("confidence") is not None:
                try:
                    confidence_delta = float(curr_conf) - float(prev["confidence"])
                except (TypeError, ValueError):
                    pass

        if len(all_ranks) < 2:
            result[ticker] = {"classification": "insufficient_data", "rank_delta": delta, "stability_index": None, "alpha_delta": alpha_delta, "confidence_delta": confidence_delta}
            continue

        std = float(np.std(all_ranks))
        trend = None
        if len(all_ranks) >= 3:
            x = np.arange(len(all_ranks))
            slope = np.polyfit(x, all_ranks, 1)[0]
            if slope < -3:
                trend = "improving"
            elif slope > 3:
                trend = "degrading"

        if trend:
            classification = trend
        elif std <= 5:
            classification = "stable"
        elif std <= 15:
            classification = "moderate"
        else:
            classification = "volatile"

        result[ticker] = {
            "classification": classification,
            "rank_delta": int(delta) if delta is not None else None,
            "stability_index": round(std, 2),
            "alpha_delta": round(alpha_delta, 4) if alpha_delta is not None else None,
            "confidence_delta": round(confidence_delta, 4) if confidence_delta is not None else None,
        }

    return result


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
    Classify signal: new (no prior run) | stable | moderate | volatile | rapidly improving | rapidly declining.
    Uses rank_delta when available (abs delta: <=5 stable, 6-15 moderate, >15 volatile); otherwise rank_std.
    """
    if rank_delta is None or (isinstance(rank_delta, float) and np.isnan(rank_delta)):
        return "new"
    if rank_delta == rank_delta:  # not nan
        if rank_delta >= RAPID_IMPROVE:
            return "rapidly improving"
        if rank_delta <= RAPID_DECLINE:
            return "rapidly declining"
        abs_delta = abs(int(rank_delta))
        if abs_delta <= 5:
            return "stable"
        if abs_delta <= 15:
            return "moderate"
        return "volatile"
    if np.isnan(rank_std) or rank_std <= STABILITY_LOW_STD:
        return "stable"
    if rank_std <= STABILITY_MED_STD:
        return "moderate"
    return "volatile"


def _commentary_from_stability(ticker: str, classification: str, rank_delta: Optional[int], stability_index: Optional[float]) -> str:
    """Human-readable commentary from DB-driven stability."""
    if classification == "new":
        return "First time in ranking — no history available"
    if classification == "returning":
        return "Previously ranked but absent from recent runs"
    parts = []
    if rank_delta is not None:
        if rank_delta < -5:
            parts.append(f"Jumped up {abs(rank_delta)} ranks since last run")
        elif rank_delta > 5:
            parts.append(f"Dropped {rank_delta} ranks since last run")
        elif rank_delta != 0:
            parts.append(f"Moved {rank_delta:+d} ranks")
        else:
            parts.append("Unchanged since last run")
    if stability_index is not None:
        parts.append(f"Rank std over recent runs: {stability_index:.1f}")
    if classification in ("improving", "degrading"):
        parts.append(f"Trend: consistently {classification}")
    return " | ".join(parts) if parts else classification


def add_ranking_insights(
    current_df: pd.DataFrame,
    previous_snapshot: Optional[dict] = None,
    history_by_ticker: Optional[dict] = None,
    db_path: Optional[str] = None,
    n_runs: int = 5,
) -> pd.DataFrame:
    """
    Enrich current_df with rank_delta, stability_index, movement_classification, ranking_commentary.

    When db_path is set: uses compute_stability_from_history (persistent, survives app restart).
    Otherwise: uses previous_snapshot and history_by_ticker (legacy).
    """
    if db_path:
        stability = compute_stability_from_history(current_df, db_path, n_runs=n_runs)
        out = current_df.copy()
        out["rank_delta"] = out["ticker"].map(lambda t: stability.get(t, {}).get("rank_delta"))
        out["stability_index"] = out["ticker"].map(lambda t: stability.get(t, {}).get("stability_index"))
        out["alpha_delta"] = out["ticker"].map(lambda t: stability.get(t, {}).get("alpha_delta"))
        out["confidence_delta"] = out["ticker"].map(lambda t: stability.get(t, {}).get("confidence_delta"))
        out["movement_classification"] = out["ticker"].map(
            lambda t: stability.get(t, {}).get("classification", "unknown")
        )
        out["ranking_commentary"] = out["ticker"].map(
            lambda t: _commentary_from_stability(
                t,
                stability.get(t, {}).get("classification", "unknown"),
                stability.get(t, {}).get("rank_delta"),
                stability.get(t, {}).get("stability_index"),
            )
        )
        # Backfill prev_rank from last run in DB for display
        out["prev_rank"] = np.nan
        if previous_snapshot:
            for idx, row in out.iterrows():
                t = row.get("ticker")
                prev = previous_snapshot.get(t)
                if prev and prev.get("rank_position") is not None:
                    out.at[idx, "prev_rank"] = prev["rank_position"]
        return out

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

        rank_std = np.nan
        if history_by_ticker and ticker in history_by_ticker:
            hist = history_by_ticker[ticker]
            positions = [h.get("rank_position") for h in hist if h.get("rank_position") is not None]
            if len(positions) >= 2:
                rank_std = stability_index(positions)
        out.at[idx, "stability_index"] = rank_std

        classification = movement_classification(rank_delta, rank_std)
        out.at[idx, "movement_classification"] = classification

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
