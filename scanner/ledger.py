"""Paper ledger: one CSV row per pick, with pure state transitions.

Lifecycle: pending (picked, waiting for the next open) -> open (filled at that open)
-> closed (stop, sell signal or time exit). Picks that size below the minimum order
or cannot be sized are recorded as skipped and never opened.

All dates are stored as ISO 'YYYY-MM-DD' strings so the CSV round-trips exactly and
the idempotency key (pick_date, ticker, config_version) always matches.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scanner.sizing import size_position, stop_price
from trader.indicators import sma

COLUMNS = [
    "pick_date", "ticker", "config_version", "status", "entry_date", "entry_price",
    "stop", "quantity", "took_it", "score_snapshot", "data_timestamp", "exit_date",
    "exit_price", "exit_reason", "pnl", "outcome", "pending_exit",
]
_STR_COLS = [
    "pick_date", "ticker", "config_version", "status", "entry_date", "score_snapshot",
    "data_timestamp", "exit_date", "exit_reason", "outcome", "pending_exit",
]
_FLOAT_COLS = ["entry_price", "stop", "quantity", "exit_price", "pnl"]
_KEY = ["pick_date", "ticker", "config_version"]

SMA_EXIT = 50
HOLD_DAYS_MAX = 56  # matches gates.hold_days_max in scanner.yaml


def _day(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=COLUMNS).reset_index(drop=True).copy()
    for c in _STR_COLS:
        df[c] = df[c].map(lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v))
                          else str(v)).astype(object)
    for c in _FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["took_it"] = df["took_it"].map(
        lambda v: str(v).strip().lower() in ("true", "1") if not isinstance(v, (bool, np.bool_))
        else bool(v)).astype(bool)
    return df


def _empty() -> pd.DataFrame:
    return _coerce(pd.DataFrame(columns=COLUMNS))


def read_ledger(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, dtype={c: str for c in _STR_COLS + ["took_it"]},
                     keep_default_na=False, na_values={c: [""] for c in _FLOAT_COLS})
    return _coerce(df)


def write_ledger(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce(df).to_csv(path, index=False)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def add_picks(ledger: pd.DataFrame, picks: pd.DataFrame, pick_date, version: str,
              equity: float) -> pd.DataFrame:
    """Append one row per pick unless (pick_date, ticker, version) already exists."""
    ledger = _coerce(ledger)
    day, version = _day(pick_date), str(version)
    existing = set(map(tuple, ledger[_KEY].to_numpy().tolist()))
    new_rows = []
    for ticker, pick in picks.iterrows():
        ticker = str(ticker)
        if (day, ticker, version) in existing:
            continue
        existing.add((day, ticker, version))
        close, atr = float(pick["close"]), float(pick["atr14"])
        stop = stop_price(close, atr)
        size = size_position(equity, close, stop)
        ok = size["status"] == "ok"
        snapshot = {"ticker": ticker, **{str(k): v for k, v in pick.items()}}
        ts = pick.get("data_timestamp", None)
        new_rows.append({
            "pick_date": day, "ticker": ticker, "config_version": version,
            "status": "pending" if ok else "skipped",
            "entry_date": "", "entry_price": np.nan,
            "stop": stop if ok else np.nan,
            "quantity": size["quantity"] if ok else np.nan,
            "took_it": False,
            "score_snapshot": json.dumps(_jsonable(snapshot), sort_keys=True),
            "data_timestamp": day if ts is None or pd.isna(ts) else str(ts),
            "exit_date": "", "exit_price": np.nan,
            "exit_reason": "" if ok else size["status"],
            "pnl": np.nan, "outcome": "", "pending_exit": "",
        })
    if not new_rows:
        return ledger
    return _coerce(pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True))


def fill_pending(ledger: pd.DataFrame, prices: dict, as_of) -> pd.DataFrame:
    """Open each pending row at the Open of the first bar after pick_date (<= as_of).
    Quantity and stop stay as sized at pick time."""
    ledger = _coerce(ledger)
    cutoff = pd.Timestamp(as_of)
    for i in ledger.index[ledger["status"] == "pending"]:
        df = prices.get(ledger.at[i, "ticker"])
        if df is None or df.empty:
            continue
        after = df[(df.index > pd.Timestamp(ledger.at[i, "pick_date"])) & (df.index <= cutoff)]
        after = after.dropna(subset=["Open"])
        if after.empty:
            continue
        ledger.at[i, "status"] = "open"
        ledger.at[i, "entry_date"] = _day(after.index[0])
        ledger.at[i, "entry_price"] = float(after["Open"].iloc[0])
    return ledger


def _replay(row: pd.Series, df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Walk bars from entry to cutoff and return the deterministic end state:
    {"closed": (date, price, reason) | None, "pending": str, "flag": (date, close) | None}."""
    entry = pd.Timestamp(row["entry_date"])
    stop = float(row["stop"])
    sma50 = sma(df["Close"], SMA_EXIT)
    window = df[(df.index >= entry) & (df.index <= cutoff)]
    pending, flag = "", None
    for date, bar in window.iterrows():
        if pending:
            return {"closed": (date, float(bar["Open"]), pending), "pending": pending,
                    "flag": flag}
        if float(bar["Low"]) <= stop:
            return {"closed": (date, min(stop, float(bar["Open"])), "stop"), "pending": "",
                    "flag": None}
        close, avg = float(bar["Close"]), sma50.loc[date]
        if pd.notna(avg) and close < float(avg):
            pending, flag = "sell_signal", (date, close)
        elif (date - entry).days >= HOLD_DAYS_MAX:
            pending, flag = "time_exit", (date, close)
    return {"closed": None, "pending": pending, "flag": flag}


def process_exits(ledger: pd.DataFrame, prices: dict, as_of) -> tuple[pd.DataFrame, list[dict]]:
    """Apply stop / sell-signal / time-exit rules to open rows up to as_of.
    Events list only transitions new in this call, so a re-run emits nothing."""
    ledger = _coerce(ledger)
    cutoff = pd.Timestamp(as_of)
    events: list[dict] = []
    for i in ledger.index[ledger["status"] == "open"]:
        row = ledger.loc[i]
        df = prices.get(row["ticker"])
        if df is None or df.empty:
            continue
        state = _replay(row, df, cutoff)
        base = {"ticker": row["ticker"], "pick_date": row["pick_date"]}
        stored = row["pending_exit"]
        if state["pending"] and state["pending"] != stored:
            fdate, fclose = state["flag"]
            events.append({**base, "kind": state["pending"], "price": fclose,
                           "date": _day(fdate)})
        if state["closed"] is None:
            ledger.at[i, "pending_exit"] = state["pending"]
            continue
        date, price, reason = state["closed"]
        pnl = (price - float(row["entry_price"])) * float(row["quantity"])
        ledger.at[i, "status"] = "closed"
        ledger.at[i, "exit_date"] = _day(date)
        ledger.at[i, "exit_price"] = price
        ledger.at[i, "exit_reason"] = reason
        ledger.at[i, "pnl"] = pnl
        ledger.at[i, "outcome"] = "win" if pnl > 0 else "loss"
        ledger.at[i, "pending_exit"] = ""
        if reason == "stop":
            events.append({**base, "kind": "stop", "price": price, "date": _day(date)})
        else:
            events.append({**base, "kind": "closed", "exit_reason": reason, "price": price,
                           "date": _day(date)})
    return ledger, events


def mark_took_it(ledger: pd.DataFrame, ticker: str, pick_date) -> pd.DataFrame:
    ledger = _coerce(ledger)
    match = (ledger["ticker"] == str(ticker)) & (ledger["pick_date"] == _day(pick_date))
    ledger.loc[match, "took_it"] = True
    return ledger


def equity(ledger: pd.DataFrame, start_equity: float, prices: dict) -> float:
    """Start equity + realised P&L + unrealised P&L of open rows at their last Close.
    An open row with no price data contributes no unrealised P&L."""
    ledger = _coerce(ledger)
    total = float(start_equity) + float(ledger.loc[ledger["status"] == "closed", "pnl"].sum())
    for _, row in ledger[ledger["status"] == "open"].iterrows():
        df = prices.get(row["ticker"])
        if df is None or df.empty:
            continue
        closes = df["Close"].dropna()
        if closes.empty:
            continue
        total += float(row["quantity"]) * (float(closes.iloc[-1]) - float(row["entry_price"]))
    return total


def summary_by_version(ledger: pd.DataFrame) -> pd.DataFrame:
    """Closed-trade stats per config_version; versions are never pooled."""
    cols = ["config_version", "trades", "wins", "losses", "win_rate", "total_pnl"]
    closed = _coerce(ledger)
    closed = closed[closed["status"] == "closed"]
    rows = []
    for version, g in closed.groupby("config_version", sort=True):
        trades = len(g)
        wins = int((g["outcome"] == "win").sum())
        rows.append({"config_version": version, "trades": trades, "wins": wins,
                     "losses": trades - wins, "win_rate": wins / trades,
                     "total_pnl": float(g["pnl"].sum())})
    return pd.DataFrame(rows, columns=cols)
