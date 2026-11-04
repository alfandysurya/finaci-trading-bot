"""Position sizing and portfolio risk rules. Pure arithmetic, no I/O."""

import math

_EPS = 1e-9


def stop_price(entry: float, atr14: float, mult: float = 2.0) -> float:
    """Stop = entry - mult * ATR14. NaN in, NaN out."""
    return entry - mult * atr14


def _invalid() -> dict:
    return {"quantity": 0.0, "value": 0.0, "risk_amount": 0.0, "status": "invalid"}


def size_position(
    equity: float,
    entry: float,
    stop: float,
    risk_pct: float = 0.01,
    cap_pct: float = 0.20,
    min_order: float = 30.0,
) -> dict:
    """Size so a stop-out loses risk_pct of equity, capped at cap_pct of equity."""
    if not all(math.isfinite(x) for x in (equity, entry, stop)):
        return _invalid()
    if equity <= 0 or entry <= 0 or stop <= 0 or stop >= entry:
        return _invalid()

    quantity = (equity * risk_pct) / (entry - stop)
    if quantity * entry > equity * cap_pct:
        quantity = (equity * cap_pct) / entry
    value = quantity * entry
    risk_amount = quantity * (entry - stop)
    status = "below_minimum" if value < min_order else "ok"
    return {
        "quantity": quantity,
        "value": value,
        "risk_amount": risk_amount,
        "status": status,
    }


def open_risk_ok(
    equity: float,
    open_risks: list[float],
    new_risk: float,
    limit_pct: float = 0.05,
) -> bool:
    """True if total open risk including new_risk stays within limit_pct of equity."""
    if not math.isfinite(equity) or equity <= 0:
        return False
    total = sum(open_risks) + new_risk
    return total <= equity * limit_pct + _EPS


def buys_paused(equity: float, peak_equity: float, pause_pct: float = 0.20) -> bool:
    """True when equity has drawn down pause_pct or more from peak."""
    if not math.isfinite(peak_equity) or peak_equity <= 0:
        return False
    return equity <= peak_equity * (1 - pause_pct) + _EPS
