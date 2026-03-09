"""
Sell signal engine: generates SELL alerts when exit conditions are met.

How the system determines sell decisions:
- Target price reached: current_price >= target_price (take profit).
- Stop loss triggered: current_price <= stop_loss (limit downside).
- Holding horizon exceeded: days_held >= holding_horizon_days (strategy exit).
- Model confidence deteriorates significantly: current confidence below threshold vs entry.
- Alpha score becomes negative: model no longer supports the position.

Sell alerts include the reason and key metrics (entry, current, return %) for the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

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
    Evaluate one holding and return a list of sell alerts (one per triggered condition).
    No alerts are generated when strategy_type is DONT_SELL (hold; ignore target, stop, horizon, confidence, alpha).

    holding: dict with entry_price/avg_price, target_price, stop_loss,
             holding_horizon_days, entry_date/added_at, confidence, alpha_score, units, strategy_type, etc.
    current_price: if None, use holding['current_price'] or holding['avg_price'].
    current_model_consensus: reliability_score or model_agreement_score from current model run;
                             sell when this is below model_consensus_negative_threshold (e.g. 0 = negative).
    """
    if (holding.get("strategy_type") or "").upper() == "DONT_SELL":
        return []
    alerts: List[SellAlert] = []
    entry = holding.get("entry_price") or holding.get("avg_price") or 0
    price = current_price if current_price is not None else (holding.get("current_price") or entry)
    if not price or price <= 0:
        return alerts

    ret_pct = _return_pct(entry, price)
    ticker = holding.get("ticker", "")

    # 1. Target price reached
    target = holding.get("target_price")
    if target is not None and target > 0 and price >= target:
        alerts.append(SellAlert(
            ticker=ticker,
            reason="Target price reached",
            entry_price=entry,
            current_price=price,
            return_pct=ret_pct,
            extra={"target_price": target},
        ))
        return alerts  # One reason per alert; first wins for display

    # 2. Stop loss triggered
    stop = holding.get("stop_loss")
    if stop is not None and stop > 0 and price <= stop:
        alerts.append(SellAlert(
            ticker=ticker,
            reason="Stop loss triggered",
            entry_price=entry,
            current_price=price,
            return_pct=ret_pct,
            extra={"stop_loss": stop},
        ))
        return alerts

    # 3. Holding horizon exceeded
    horizon_days = holding.get("holding_horizon_days")
    if horizon_days is not None and horizon_days > 0:
        added = holding.get("entry_date") or holding.get("added_at")
        if added:
            try:
                if isinstance(added, str):
                    dt = datetime.strptime(added[:10], "%Y-%m-%d")
                else:
                    dt = added
                days_held = (datetime.now() - dt).days if isinstance(dt, datetime) else 0
                if days_held >= horizon_days:
                    alerts.append(SellAlert(
                        ticker=ticker,
                        reason="Holding horizon exceeded",
                        entry_price=entry,
                        current_price=price,
                        return_pct=ret_pct,
                        extra={"holding_horizon_days": horizon_days, "days_held": days_held},
                    ))
                    return alerts
            except Exception:
                pass

    # 4. Model confidence deteriorated significantly
    entry_conf = holding.get("confidence")
    if entry_conf is not None and current_confidence is not None:
        if current_confidence < entry_conf - confidence_deterioration_threshold:
            alerts.append(SellAlert(
                ticker=ticker,
                reason="Model confidence deteriorated significantly",
                entry_price=entry,
                current_price=price,
                return_pct=ret_pct,
                confidence=current_confidence,
                extra={"entry_confidence": entry_conf},
            ))
            return alerts

    # 5. Alpha score becomes negative
    if current_alpha_score is not None and current_alpha_score < 0:
        alerts.append(SellAlert(
            ticker=ticker,
            reason="Alpha score became negative",
            entry_price=entry,
            current_price=price,
            return_pct=ret_pct,
            alpha_score=current_alpha_score,
        ))
        return alerts

    # 6. Model consensus turns negative (reliability / agreement below threshold)
    if current_model_consensus is not None and current_model_consensus < model_consensus_negative_threshold:
        alerts.append(SellAlert(
            ticker=ticker,
            reason="Model consensus turned negative",
            entry_price=entry,
            current_price=price,
            return_pct=ret_pct,
            extra={"model_consensus_score": current_model_consensus},
        ))

    return alerts


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


def get_all_sell_alerts(
    holdings: List[dict],
    model_results: Optional[pd.DataFrame] = None,
    only_open: bool = True,
) -> List[SellAlert]:
    """
    Evaluate all holdings and return a flat list of sell alerts.

    If model_results is provided, current_confidence and current_alpha_score
    are looked up by ticker.
    Holdings with strategy_type DONT_SELL never generate sell alerts.
    """
    out: List[SellAlert] = []
    for h in holdings:
        if only_open and (h.get("status") or "OPEN") != "OPEN":
            continue
        if (h.get("strategy_type") or "").upper() == "DONT_SELL":
            continue
        conf = alpha = consensus = None
        if model_results is not None and not model_results.empty and "ticker" in model_results.columns:
            m = model_results[model_results["ticker"] == h["ticker"]]
            if not m.empty:
                conf = m.iloc[0].get("confidence")
                if pd.notna(conf):
                    conf = float(conf)
                a = m.iloc[0].get("alpha_score")
                if pd.isna(a):
                    a = m.iloc[0].get("predicted_return_pct")
                    if pd.notna(a):
                        a = float(a) / 100.0
                if pd.notna(a):
                    alpha = float(a)
                for col in ("reliability_score", "model_agreement_score", "model_consensus_score"):
                    if col in m.columns and pd.notna(m.iloc[0].get(col)):
                        consensus = float(m.iloc[0][col])
                        break
        alerts = evaluate_sell_signals(
            h, current_confidence=conf, current_alpha_score=alpha, current_model_consensus=consensus
        )
        out.extend(alerts)
    return out
