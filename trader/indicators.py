"""Indicator signal library.

Each family turns an OHLCV frame into a *target position* series in {0, 1}, decided on
the close of each day (long-only: 1 = hold the stock, 0 = sit in cash). The backtest
shifts it so trades happen at the next open — no look-ahead.

Every family declares a small parameter grid. Keep grids small on purpose: the more
combinations we test, the more likely the "winner" is luck (multiple testing).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd


# ---------- building blocks ----------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def _state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Stateful long/flat position from entry and exit events (exit wins on ties)."""
    pos = pd.Series(np.nan, index=entry.index)
    pos[entry.fillna(False).astype(bool)] = 1.0
    pos[exit_.fillna(False).astype(bool)] = 0.0
    return pos.ffill().fillna(0.0)


def _flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    return cond.where(valid.notna(), False).astype(float)


# ---------- signal families ----------

def sma_cross(df, fast, slow):
    f, s = sma(df.Close, fast), sma(df.Close, slow)
    return _flag(f > s, s)


def ema_cross(df, fast, slow):
    f, s = ema(df.Close, fast), ema(df.Close, slow)
    return _flag(f > s, s)


def price_above_sma(df, n):
    m = sma(df.Close, n)
    return _flag(df.Close > m, m)


def macd(df, fast, slow, signal):
    line = ema(df.Close, fast) - ema(df.Close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return _flag(line > sig, sig)


def rsi_reversion(df, n, low, high):
    """Buy oversold, sell when it recovers. Only in an uptrend (close > 200d SMA)."""
    r = rsi(df.Close, n)
    trend = df.Close > sma(df.Close, 200)
    return _state((r < low) & trend, (r > high) | ~trend)


def donchian_breakout(df, entry, exit):
    upper = df.High.rolling(entry, min_periods=entry).max().shift(1)
    lower = df.Low.rolling(exit, min_periods=exit).min().shift(1)
    return _state(df.Close > upper, df.Close < lower)


def momentum(df, lookback, skip):
    """Time-series momentum: long when the return from t-lookback to t-skip is positive."""
    past = df.Close.shift(skip) / df.Close.shift(lookback) - 1
    return _flag(past > 0, past)


@dataclass(frozen=True)
class Family:
    name: str
    fn: Callable[..., pd.Series]
    grid: dict[str, list]
    valid: Callable[[dict], bool] = lambda p: True

    def configs(self) -> list[dict]:
        keys = list(self.grid)
        out = []
        for vals in product(*self.grid.values()):
            p = dict(zip(keys, vals))
            if self.valid(p):
                out.append(p)
        return out


FAMILIES: dict[str, Family] = {f.name: f for f in [
    Family("sma_cross", sma_cross, {"fast": [10, 20, 50], "slow": [50, 100, 200]},
           lambda p: p["fast"] < p["slow"]),
    Family("ema_cross", ema_cross, {"fast": [12, 26, 50], "slow": [50, 100, 200]},
           lambda p: p["fast"] < p["slow"]),
    Family("price_above_sma", price_above_sma, {"n": [50, 100, 150, 200]}),
    Family("macd", macd, {"fast": [12], "slow": [26], "signal": [9]}),
    Family("rsi_reversion", rsi_reversion, {"n": [2, 14], "low": [10, 30], "high": [50, 70]},
           lambda p: (p["n"] == 2 and p["low"] == 10) or (p["n"] == 14 and p["low"] == 30)),
    Family("donchian_breakout", donchian_breakout, {"entry": [20, 55], "exit": [10, 20]},
           lambda p: p["exit"] < p["entry"]),
    Family("momentum", momentum, {"lookback": [126, 252], "skip": [0, 21]}),
]}


def all_configs() -> list[tuple[str, dict]]:
    return [(name, p) for name, fam in FAMILIES.items() for p in fam.configs()]


def config_id(name: str, params: dict) -> str:
    return name + "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"


def target_position(df: pd.DataFrame, name: str, params: dict) -> pd.Series:
    return FAMILIES[name].fn(df, **params).astype(float)


def snapshot(df: pd.DataFrame) -> dict:
    """Latest indicator readings, used to explain proposals to a human."""
    c = df.Close
    last = lambda s, d=2: None if pd.isna(s.iloc[-1]) else round(float(s.iloc[-1]), d)
    return {
        "close": last(c),
        "sma50": last(sma(c, 50)),
        "sma200": last(sma(c, 200)),
        "rsi14": last(rsi(c, 14), 1),
        "ret_1m": last(c / c.shift(21) - 1, 4),
        "ret_6m": last(c / c.shift(126) - 1, 4),
    }
