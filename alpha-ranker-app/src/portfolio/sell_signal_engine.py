"""
Sell signal engine: generates SELL/REVIEW/HOLD signals using model predictions and rules.

- Model-driven (active mode): predicted_return_pct, alpha_rank, model_agreement_score drive signals.
- Stop-loss and target apply in all modes; passive mode only emits SELL on stop-loss/target.
- Sell mode per strategy (portfolio_settings): disabled | passive | active.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SellAlert:
    ticker: str
    reason: str
    entry_price: float
    current_price: float
    return_pct: float
    confidence: Optional[float] = None
    alpha_score: Optional[float] = None
    extra: Optional[dict] = None


def _return_pct(entry: float, current: float) -> float:
    if not entry or entry <= 0:
        return 0.0
    return (current / entry - 1) * 100


def _format_model_note(model_view: Optional[Dict[str, Any]]) -> str:
    """Human-readable summary of model predictions for UI."""
    if not model_view:
        return "No model data"
    parts = []
    pr = model_view.get("predicted_return_pct")
    if pr is not None and pd.notna(pr):
        parts.append(f"Pred: {float(pr):+.1f}%")
    rank = model_view.get("alpha_rank")
    if rank is not None and pd.notna(rank):
        parts.append(f"Rank #{int(rank)}")
    agr = model_view.get("model_agreement")
    if agr is not None and pd.notna(agr):
        parts.append(f"Agr: {float(agr)*100:.0f}%")
    conf = model_view.get("confidence")
    if conf is not None and pd.notna(conf):
        parts.append(f"Conf: {float(conf):+.2f}")
    return " | ".join(parts) if parts else "No model data"


def _get_model_view(
    ticker: str,
    model_results: Optional[pd.DataFrame],
    n_stocks: int,
) -> Dict[str, Any]:
    if model_results is None or model_results.empty or "ticker" not in model_results.columns:
        return {}
    match = model_results[model_results["ticker"] == ticker]
    if match.empty:
        return {}
    row = match.iloc[0]
    rank = row.get("alpha_rank")
    if pd.isna(rank):
        rank = None
    else:
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            rank = None
    rank_pct = (rank / n_stocks) if (rank is not None and n_stocks > 0) else None
    agr = row.get("model_agreement_score")
    if pd.notna(agr):
        agr = float(agr)
    else:
        agr = None
    pr = row.get("predicted_return_pct")
    if pd.notna(pr):
        pr = float(pr)
    else:
        pr = None
    conf = row.get("confidence")
    if pd.notna(conf):
        conf = float(conf)
    else:
        conf = None
    raw = row.get("alpha_score_raw")
    if pd.notna(raw):
        raw = float(raw)
    else:
        raw = None
    rel = row.get("reliability_score")
    if pd.notna(rel):
        rel = float(rel)
    else:
        rel = None
    return {
        "predicted_return_pct": pr,
        "alpha_rank": rank,
        "alpha_rank_pct": rank_pct,
        "alpha_score_raw": raw,
        "confidence": conf,
        "model_agreement": agr,
        "reliability": rel,
    }


def get_all_sell_signals(
    holdings: List[dict],
    model_results: Optional[pd.DataFrame] = None,
    only_open: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return one signal dict per (open) holding for UI table.
    Each dict: ticker, signal (SELL|REVIEW|HOLD), reason, reasons, trigger, urgency, strategy,
    model_view, model_note, entry_price, current_price, return_pct, action.
    """
    try:
        from core.engine_config import get_portfolio_settings
        ps = get_portfolio_settings()
    except Exception:
        ps = {}
    default_stop_pct = float(ps.get("default_stop_loss_pct", 10.0))
    n_stocks = len(model_results) if model_results is not None and not model_results.empty else 1
    out: List[Dict[str, Any]] = []

    for h in holdings:
        if only_open and (h.get("status") or "OPEN") != "OPEN":
            continue
        ticker = h.get("ticker", "")
        strategy = (h.get("strategy_type") or "LONG_TERM").upper().replace("-", "_")
        sell_mode_key = f"sell_mode_{strategy.lower()}"
        sell_mode = (ps.get(sell_mode_key) or ("disabled" if strategy == "DONT_SELL" else "active")).lower()

        entry = h.get("entry_price") or h.get("avg_price") or 0
        price = h.get("current_price") or entry
        if not price or price <= 0:
            price = entry
        return_pct = _return_pct(entry, price)
        model_view = _get_model_view(ticker, model_results, n_stocks)
        model_note = _format_model_note(model_view)

        # Disabled: always HOLD, no sell logic
        if sell_mode == "disabled" or strategy == "DONT_SELL":
            out.append({
                "ticker": ticker,
                "signal": "HOLD",
                "reason": "Sell signals disabled for " + (strategy if strategy != "DONT_SELL" else "DONT_SELL"),
                "reasons": [],
                "trigger": None,
                "urgency": None,
                "strategy": strategy,
                "model_view": model_view,
                "model_note": model_note,
                "entry_price": entry,
                "current_price": price,
                "return_pct": return_pct,
                "action": "none",
                "predicted_return_pct": model_view.get("predicted_return_pct"),
                "alpha_rank": model_view.get("alpha_rank"),
                "model_agreement": model_view.get("model_agreement"),
            })
            continue

        reasons: List[str] = []
        trigger: Optional[str] = None

        # Stop-loss: by price or by return %
        stop_price = h.get("stop_loss")
        if stop_price is not None and stop_price > 0 and price <= stop_price:
            reasons.append(f"Stop-loss hit: price {price:.2f} <= {stop_price:.2f}")
            trigger = "stop_loss"
        elif return_pct <= -default_stop_pct:
            reasons.append(f"Stop-loss hit: {return_pct:+.1f}% (limit -{default_stop_pct:.0f}%)")
            trigger = "stop_loss"

        # Target
        target = h.get("target_price")
        if target is not None and target > 0 and price >= target and trigger is None:
            reasons.append("Target price reached")
            trigger = "target"

        # Passive: only stop-loss/target trigger SELL
        if sell_mode == "passive":
            if trigger in ("stop_loss", "target"):
                out.append({
                    "ticker": ticker,
                    "signal": "SELL",
                    "reason": " | ".join(reasons),
                    "reasons": reasons,
                    "trigger": trigger,
                    "urgency": "high",
                    "strategy": strategy,
                    "model_view": model_view,
                    "model_note": model_note,
                    "entry_price": entry,
                    "current_price": price,
                    "return_pct": return_pct,
                    "action": "sell",
                    "predicted_return_pct": model_view.get("predicted_return_pct"),
                    "alpha_rank": model_view.get("alpha_rank"),
                    "model_agreement": model_view.get("model_agreement"),
                })
            else:
                out.append({
                    "ticker": ticker,
                    "signal": "HOLD",
                    "reason": f"Passive ({strategy}): no stop-loss triggered",
                    "reasons": [],
                    "trigger": None,
                    "urgency": None,
                    "strategy": strategy,
                    "model_view": model_view,
                    "model_note": model_note,
                    "entry_price": entry,
                    "current_price": price,
                    "return_pct": return_pct,
                    "action": "none",
                    "predicted_return_pct": model_view.get("predicted_return_pct"),
                    "alpha_rank": model_view.get("alpha_rank"),
                    "model_agreement": model_view.get("model_agreement"),
                })
            continue

        # Active: add model-based reasons
        pr = model_view.get("predicted_return_pct")
        rank_pct = model_view.get("alpha_rank_pct")
        agreement = model_view.get("model_agreement")

        if pr is not None and pr < 0 and trigger is None:
            reasons.append(f"Model predicts negative return: {pr:+.1f}%")
            trigger = "model_negative"
        if rank_pct is not None and rank_pct > 0.80 and trigger is None:
            reasons.append(f"Alpha rank degraded: bottom {(1 - rank_pct) * 100:.0f}% of universe")
            trigger = "rank_degraded"
        if agreement is not None and agreement < 0.3 and trigger is None:
            reasons.append(f"Model disagreement: {agreement * 100:.0f}% of models bullish")
            trigger = "model_disagreement"

        # Horizon exceeded
        horizon_days = h.get("holding_horizon_days")
        if horizon_days is not None and horizon_days > 0 and trigger is None:
            added = h.get("entry_date") or h.get("added_at")
            if added:
                try:
                    if isinstance(added, str):
                        dt = datetime.strptime(added[:10], "%Y-%m-%d")
                    else:
                        dt = added
                    days_held = (datetime.now() - dt).days if isinstance(dt, datetime) else 0
                    if days_held >= horizon_days:
                        reasons.append(f"Holding horizon exceeded ({days_held}d >= {horizon_days}d)")
                        trigger = "horizon"
                except Exception:
                    pass

        # Confidence drop
        entry_conf = h.get("confidence")
        curr_conf = model_view.get("confidence")
        if entry_conf is not None and curr_conf is not None and trigger is None:
            if curr_conf < entry_conf - 0.5:
                reasons.append("Model confidence deteriorated significantly")
                trigger = "confidence"

        # Determine signal strength
        if reasons:
            if trigger == "stop_loss":
                signal_strength = "SELL"
                urgency = "high"
            elif trigger == "target":
                signal_strength = "SELL"
                urgency = "high"
            elif len(reasons) >= 2:
                signal_strength = "SELL"
                urgency = "medium"
            else:
                signal_strength = "REVIEW"
                urgency = "low"
            out.append({
                "ticker": ticker,
                "signal": signal_strength,
                "reason": " | ".join(reasons),
                "reasons": reasons,
                "trigger": trigger,
                "urgency": urgency,
                "strategy": strategy,
                "model_view": model_view,
                "model_note": model_note,
                "entry_price": entry,
                "current_price": price,
                "return_pct": return_pct,
                "action": "sell" if signal_strength == "SELL" else "review",
                "predicted_return_pct": model_view.get("predicted_return_pct"),
                "alpha_rank": model_view.get("alpha_rank"),
                "model_agreement": model_view.get("model_agreement"),
            })
        else:
            hold_reason = f"Model predicts {pr:+.1f}%" if pr is not None else "Model outlook positive"
            out.append({
                "ticker": ticker,
                "signal": "HOLD",
                "reason": hold_reason,
                "reasons": [],
                "trigger": None,
                "urgency": None,
                "strategy": strategy,
                "model_view": model_view,
                "model_note": model_note,
                "entry_price": entry,
                "current_price": price,
                "return_pct": return_pct,
                "action": "none",
                "predicted_return_pct": model_view.get("predicted_return_pct"),
                "alpha_rank": model_view.get("alpha_rank"),
                "model_agreement": model_view.get("model_agreement"),
            })

    return out


def get_all_sell_alerts(
    holdings: List[dict],
    model_results: Optional[pd.DataFrame] = None,
    only_open: bool = True,
) -> List[SellAlert]:
    """
    Backward-compat: return list of SellAlert for SELL/REVIEW only (for legacy text display).
    """
    signals = get_all_sell_signals(holdings, model_results, only_open)
    alerts: List[SellAlert] = []
    for s in signals:
        if s["signal"] not in ("SELL", "REVIEW"):
            continue
        alerts.append(SellAlert(
            ticker=s["ticker"],
            reason=s["reason"],
            entry_price=s["entry_price"],
            current_price=s["current_price"],
            return_pct=s["return_pct"],
            confidence=s.get("model_view", {}).get("confidence"),
            alpha_score=s.get("model_view", {}).get("alpha_score_raw"),
            extra={"trigger": s.get("trigger"), "urgency": s.get("urgency")},
        ))
    return alerts


def evaluate_sell_signals(
    holding: dict,
    current_price: Optional[float] = None,
    current_confidence: Optional[float] = None,
    current_alpha_score: Optional[float] = None,
    current_model_consensus: Optional[float] = None,
    confidence_deterioration_threshold: float = 0.5,
    model_consensus_negative_threshold: float = 0.0,
) -> List[SellAlert]:
    """
    Evaluate one holding; returns list of sell alerts (legacy single-holding API).
    Prefer get_all_sell_signals() for UI table.
    """
    mr = None
    if current_confidence is not None or current_alpha_score is not None or current_model_consensus is not None:
        t = holding.get("ticker", "")
        if t:
            a_raw = (current_alpha_score / 100.0) if current_alpha_score is not None else None
            mr = pd.DataFrame({
                "ticker": [t],
                "confidence": [current_confidence],
                "alpha_score_raw": [a_raw],
                "alpha_rank": [None],
                "predicted_return_pct": [current_alpha_score if current_alpha_score is not None else None],
                "model_agreement_score": [current_model_consensus],
            })
    return get_all_sell_alerts([holding], mr, only_open=True)


def format_sell_alert(alert: SellAlert) -> str:
    """Human-readable sell alert message."""
    lines = [
        "SELL ALERT",
        f"Ticker: {alert.ticker}",
        f"Reason: {alert.reason}",
        f"Entry: {alert.entry_price:.2f}",
        f"Current: {alert.current_price:.2f}",
        f"Return: {alert.return_pct:+.1f}%",
    ]
    return "\n".join(lines)
