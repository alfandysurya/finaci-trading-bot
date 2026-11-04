"""Scanner indicators: pure pandas/numpy helpers and a latest-readings summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.indicators import ema, rsi, sma


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range with Wilder smoothing."""
    prev = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - prev).abs(), (df.Low - prev).abs()],
                   axis=1).max(axis=1, skipna=True)
    tr = tr.where(prev.notna(), df.High - df.Low)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line minus its signal line."""
    line = ema(close, fast) - ema(close, slow)
    return line - line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def volume_trend(volume: pd.Series, short: int = 20, long: int = 60) -> pd.Series:
    """Short-window mean volume divided by long-window mean volume."""
    return sma(volume, short) / sma(volume, long)


def drawdown_from_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Close relative to its rolling high (<= 0)."""
    return close / close.rolling(window, min_periods=1).max() - 1


def latest_metrics(df: pd.DataFrame) -> dict:
    """Latest reading of each metric as a float (nan when history is too short)."""
    c, v = df.Close, df.Volume
    a = atr(df, 14)
    last = lambda s: float(s.iloc[-1]) if len(s) else float("nan")
    return {
        "close": last(c),
        "sma50": last(sma(c, 50)),
        "sma200": last(sma(c, 200)),
        "rsi14": last(rsi(c, 14)),
        "macd_hist": last(macd_hist(c)),
        "vol_trend": last(volume_trend(v)),
        "ret_63d": last(c / c.shift(63) - 1),
        "atr14": last(a),
        "atr_pct": last(a / c),
        "drawdown": last(drawdown_from_high(c)),
        "vol_20d_ann": last(c.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)),
        "dollar_vol_20d": last((c * v).rolling(20, min_periods=20).mean()),
    }
