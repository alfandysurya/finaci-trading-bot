"""Single-asset long/flat backtest with next-open execution and Pluang costs.

Timing (no look-ahead):
  day t close   -> indicator decides target position
  day t+1 open  -> order fills (in practice: you confirm the proposal in Pluang)
Daily strategy return on day t+1:
  holding both days        : close[t+1] / close[t] - 1
  entered at open t+1      : close[t+1] / open[t+1] - 1
  exited at open t+1       : open[t+1] / close[t] - 1
minus cost_per_side on every entry and exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def strategy_returns(df: pd.DataFrame, target: pd.Series, cost: float) -> pd.DataFrame:
    pos = target.shift(1).fillna(0.0)          # position held after day's open
    prev = pos.shift(1).fillna(0.0)
    c, o = df.Close, df.Open
    pc = c.shift(1)

    hold = (pos == 1) & (prev == 1)
    enter = (pos == 1) & (prev == 0)
    exit_ = (pos == 0) & (prev == 1)

    r = pd.Series(0.0, index=df.index)
    r[hold] = (c / pc - 1)[hold]
    r[enter] = (c / o - 1)[enter]
    r[exit_] = (o / pc - 1)[exit_]
    trades = (pos - prev).abs()
    r = r - trades * cost
    return pd.DataFrame({"ret": r.fillna(0.0), "pos": pos, "trade": trades})


def buy_and_hold_returns(df: pd.DataFrame, cost: float) -> pd.Series:
    r = df.Close.pct_change().fillna(0.0)
    r.iloc[0] -= cost
    return r


def metrics(ret: pd.Series, pos: pd.Series | None = None, trades: pd.Series | None = None,
            rf: float = 0.0) -> dict:
    ret = ret.dropna()
    n = len(ret)
    if n < 2:
        return {}
    years = n / TRADING_DAYS
    equity = (1 + ret).cumprod()
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    excess = ret - rf / TRADING_DAYS
    sharpe = float(excess.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else 0.0
    dd = equity / equity.cummax() - 1
    mdd = float(dd.min())
    out = {
        "cagr": cagr, "total_return": total, "vol": vol, "sharpe": sharpe,
        "max_drawdown": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "years": years,
    }
    if pos is not None:
        out["exposure"] = float(pos.mean())
    if trades is not None:
        round_trips = trades.sum() / 2
        out["trades_per_year"] = float(round_trips / years)
    return out


def run(df: pd.DataFrame, target: pd.Series, cost: float, rf: float = 0.0) -> dict:
    res = strategy_returns(df, target, cost)
    m = metrics(res["ret"], res["pos"], res["trade"], rf)
    m["returns"] = res["ret"]
    return m
