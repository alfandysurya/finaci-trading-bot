"""Validate the portfolio we actually trade, and decide whether it may use real money.

Replays the live rule month by month: signals at the last close of each month, orders filled
at the next open, with the same target_weights() and order thresholds as proposals.build().
Results are shown on the trading universe and on bias_check_tickers (large caps chosen as of
2009, so less hindsight). The go-live gate requires beating SPY's Sharpe on both.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from .indicators import config_id, target_position
from .proposals import target_weights


@dataclass
class SimResult:
    returns: pd.Series
    order_dates: list
    orders_per_year: float


def simulate(prices: dict[str, pd.DataFrame], tickers: list[str], cfg: dict, cost: float,
             start: str) -> SimResult:
    pcfg, strat = cfg["portfolio"], cfg["strategy"]
    cal = prices[cfg["universe"]["benchmark"]].index
    frames = {t: prices[t].reindex(cal).ffill() for t in tickers}
    close = pd.DataFrame({t: f.Close for t, f in frames.items()})
    open_ = pd.DataFrame({t: f.Open for t, f in frames.items()})
    signal = pd.DataFrame({t: target_position(prices[t], strat["name"], strat["params"]).reindex(cal).ffill()
                           for t in tickers})
    vol = close.pct_change().rolling(pcfg["vol_lookback"]).std()
    mom = (close / close.shift(126) - 1).fillna(0.0)
    bars = close.notna().cumsum()
    month_end = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).transform("max") == cal

    cash = float(pcfg["equity_usd"])
    shares = pd.Series(0.0, index=tickers)
    days = cal[cal >= pd.Timestamp(start)]
    equity, order_dates, pending = [], [], []

    for d in days:
        for t, delta, exit_all in sorted(pending, key=lambda o: o[1]):   # sells first
            price = open_.at[d, t]
            if delta < 0:
                qty = shares[t] if exit_all else min(shares[t], -delta / price)
                shares[t] -= qty
                cash += qty * price * (1 - cost)
            else:
                usd = min(delta, cash / (1 + cost))
                shares[t] += usd / price
                cash -= usd * (1 + cost)
            order_dates.append(d)
        pending = []

        value = cash + float((shares * close.loc[d].fillna(0.0)).sum())
        equity.append(value)
        if not month_end[d]:
            continue

        ok = bars.loc[d] >= 260
        table = pd.DataFrame({"signal": signal.loc[d], "vol": vol.loc[d], "mom6m": mom.loc[d],
                              "held": shares > 0})[ok]
        weights = target_weights(table, pcfg)
        min_delta = max(pcfg["min_order_usd"], pcfg["rebalance_threshold"] * value)
        for t in set(weights.index) | set(shares[shares > 0].index):
            current = shares[t] * close.at[d, t]
            delta = float(weights.get(t, 0.0)) * value - current
            exit_all = t not in weights.index
            if exit_all or abs(delta) >= min_delta:
                pending.append((t, delta, exit_all))

    curve = pd.Series(equity, index=days)
    years = len(curve) / bt.TRADING_DAYS
    return SimResult(curve.pct_change().fillna(0.0), order_dates, len(order_dates) / years)


def go_live(results: dict[str, dict], spy: dict) -> bool:
    return all(r["sharpe"] > spy["sharpe"] for r in results.values())


def validate(prices: dict[str, pd.DataFrame], cfg: dict, cost: float, outdir: Path) -> dict:
    u = cfg["universe"]
    start = cfg["validation"]["start"]
    bench = prices[u["benchmark"]].Close
    spy = bt.metrics(bench.pct_change()[bench.index >= pd.Timestamp(start)].fillna(0.0))

    results = {}
    for label, tickers in [("trading universe", u["tickers"]), ("bias check (2009 large caps)", u["bias_check_tickers"])]:
        sim = simulate(prices, tickers, cfg, cost, start)
        results[label] = {**bt.metrics(sim.returns), "orders_per_year": sim.orders_per_year}

    approved = go_live(results, spy)
    strat = cfg["strategy"]
    lines = [f"# Portfolio validation: {'APPROVED' if approved else 'NOT APPROVED'} for real money", "",
             f"Rule: `{config_id(strat['name'], strat['params'])}`, checked monthly. After costs of {cost:.3%} per side, "
             f"from {start}. Gate: Sharpe above {u['benchmark']} on both universes.", "",
             "| portfolio | CAGR | Sharpe | max DD | orders/yr |", "|---|---|---|---|---|",
             f"| {u['benchmark']} buy & hold | {spy['cagr']:.1%} | {spy['sharpe']:.2f} | {spy['max_drawdown']:.1%} | 0 |"]
    lines += [f"| {k} | {r['cagr']:.1%} | {r['sharpe']:.2f} | {r['max_drawdown']:.1%} | {r['orders_per_year']:.0f} |"
              for k, r in results.items()]
    lines += ["", "_Both stock lists contain only companies that survived, so these results are still optimistic. "
              "Do not change the rule just to pass this gate._"]

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    out = {"go_live": approved, "benchmark": spy, "results": results}
    (outdir / "validation.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out
