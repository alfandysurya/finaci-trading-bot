"""Turn the selected strategy into today's order proposals.

Output is a JSON file (machine-readable, what Claude reads) plus a Markdown twin (what you
read). Nothing here talks to Pluang. Execution happens in a Claude conversation with the
Pluang connector, where every order still needs your confirmation in the Pluang app.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .indicators import config_id, snapshot, target_position


def load_holdings(path: Path | None) -> dict:
    """holdings.yaml: {cash_usd: 120.5, positions: {AAPL: 0.35, MSFT: 0.1}}  (shares)."""
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _cap_weights(w: pd.Series, cap: float, total: float) -> pd.Series:
    """Scale to `total`, cap each name, redistribute excess to uncapped names."""
    w = w / w.sum() * total
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w.clip(upper=cap)


def target_weights(table: pd.DataFrame, pcfg: dict) -> pd.Series:
    """Target weight per chosen ticker. Shared with simulate.py so validation matches live trading.

    Expects columns: signal, held, mom6m, vol. Keeps holdings that still signal, then strongest momentum.
    """
    cands = table[table.signal == 1]
    chosen = (cands.sort_values(["held", "mom6m"], ascending=[False, False])
              .head(pcfg["max_positions"]).index.tolist())
    if not chosen:
        return pd.Series(dtype=float)
    raw = (1 / table.loc[chosen, "vol"]) if pcfg["sizing"] == "inverse_vol" else pd.Series(1.0, index=chosen)
    return _cap_weights(raw.astype(float), pcfg["max_weight"], 1 - pcfg["cash_buffer"])


def build(prices: dict[str, pd.DataFrame], selection: dict, cfg: dict,
          holdings: dict | None = None, now: datetime | None = None, go_live: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    pcfg, ecfg = cfg["portfolio"], cfg["execution"]
    tickers = [t for t in cfg["universe"]["tickers"] if t in prices]
    name, params = selection["name"], selection["params"]
    holdings = holdings or {}
    shares = holdings.get("positions", {}) or {}

    rows = []
    for t in tickers:
        df = prices[t].dropna(subset=["Close"])
        if len(df) < 260:
            continue
        sig = target_position(df, name, params).iloc[-1]
        snap = snapshot(df)
        vol = df.Close.pct_change().tail(pcfg["vol_lookback"]).std()
        rows.append({"ticker": t, "signal": int(sig), "close": float(df.Close.iloc[-1]),
                     "vol": float(vol), "mom6m": snap["ret_6m"] or 0.0,
                     "held": float(shares.get(t, 0)) > 0, "snap": snap,
                     "as_of": df.index[-1]})
    table = pd.DataFrame(rows).set_index("ticker")
    as_of = table.as_of.max()

    # Bad data must stop the run: it would silently skip sells and misstate equity.
    missing = [t for t, qty in shares.items() if float(qty) > 0 and t not in table.index]
    if missing:
        raise ValueError(f"Held tickers have no usable price data: {', '.join(missing)}")
    stale = table.index[table.as_of < as_of].tolist()
    if stale:
        raise ValueError(f"Tickers are missing the latest bar ({as_of.date()}): {', '.join(stale)}")

    investable = 1 - pcfg["cash_buffer"]
    weights = target_weights(table, pcfg)
    chosen = weights.index.tolist()

    current_val = {t: float(shares.get(t, 0)) * table.close.get(t, np.nan) for t in shares if t in table.index}
    equity = (holdings.get("cash_usd", 0) + sum(current_val.values())) if holdings else pcfg["equity_usd"]
    min_delta = max(pcfg["min_order_usd"], pcfg["rebalance_threshold"] * equity)

    orders, targets = [], []
    for t in sorted(set(chosen) | set(current_val)):
        close = table.close[t]
        tgt_w = float(weights.get(t, 0.0))
        tgt_usd = tgt_w * equity
        cur_usd = current_val.get(t, 0.0)
        delta = tgt_usd - cur_usd
        snap = table.snap[t]
        mom = f"{snap['ret_6m']:+.1%}" if snap["ret_6m"] is not None else "n/a"
        reason = (f"{config_id(name, params)} signal={'LONG' if t in chosen else 'FLAT'}; "
                  f"close {snap['close']}, SMA50 {snap['sma50']}, SMA200 {snap['sma200']}, "
                  f"RSI14 {snap['rsi14']}, 6m {mom}")
        if t not in chosen and table.signal[t] == 1:
            reason += " (signal LONG but outside top max_positions)"
        targets.append({"ticker": t, "target_weight": round(tgt_w, 4), "target_usd": round(tgt_usd, 2),
                        "current_usd": round(cur_usd, 2)})
        exit_all = bool(tgt_w == 0 and cur_usd > 0)
        if not exit_all and abs(delta) < min_delta:
            continue
        side = "BUY" if delta > 0 else "SELL"
        buf = ecfg["limit_buffer"]
        orders.append({
            "ticker": t, "side": side,
            "usd_amount": round(abs(delta), 2),
            "sell_all": exit_all,
            "order_type": "LIMIT",
            "limit_price": round(close * (1 + buf if side == "BUY" else 1 - buf), 2),
            "ref_price": round(close, 2),
            "max_price_deviation": ecfg["max_price_deviation"],
            "reason": reason,
        })
    # sells first so their proceeds fund the buys
    orders.sort(key=lambda o: (o["side"] != "SELL", -o["usd_amount"]))

    notes = []
    if not go_live:
        notes.append("PAPER ONLY: this strategy has not passed `validate`. Do not place real orders.")
    if not holdings:
        notes.append("No holdings file given: amounts assume an empty account. Claude must reconcile "
                     "against the live Pluang portfolio before proposing anything.")
    if (now.replace(tzinfo=None) - as_of.to_pydatetime()).days > 4:
        notes.append(f"Price data is stale (last bar {as_of.date()}). Refresh data before trading.")
    if not chosen:
        notes.append("No ticker has a LONG signal: target is all cash.")

    return {
        "schema": "pluang-proposals/v1",
        "generated_at": now.isoformat(timespec="seconds"),
        "valid_until": (now + timedelta(hours=ecfg["proposal_valid_hours"])).isoformat(timespec="seconds"),
        "data_as_of": str(as_of.date()),
        "strategy": config_id(name, params),
        "equity_usd": round(equity, 2),
        "limits": {"max_positions": pcfg["max_positions"], "max_weight": pcfg["max_weight"],
                   "max_total_buy_usd": round(equity * investable, 2)},
        "orders": orders,
        "targets": targets,
        "notes": notes,
    }


def to_markdown(p: dict) -> str:
    lines = [f"# Trade proposals — data as of {p['data_as_of']}", "",
             f"Strategy: `{p['strategy']}` · Equity assumed: ${p['equity_usd']:,.2f} · "
             f"Valid until {p['valid_until']}", ""]
    if p["notes"]:
        lines += ["**Notes**", ""] + [f"- {n}" for n in p["notes"]] + [""]
    if p["orders"]:
        lines += ["| # | side | ticker | amount | limit | ref price | reason |", "|---|---|---|---|---|---|---|"]
        for i, o in enumerate(p["orders"], 1):
            amt = "ALL" if o["sell_all"] else f"${o['usd_amount']:,.2f}"
            lines.append(f"| {i} | {o['side']} | {o['ticker']} | {amt} | {o['limit_price']} | {o['ref_price']} | {o['reason']} |")
    else:
        lines.append("_No orders: portfolio already matches targets._")
    lines += ["", "Each order is confirmed by you in Pluang. Skip anything you disagree with."]
    return "\n".join(lines)


def save(p: dict, outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"proposals_{p['data_as_of']}"
    j, m = outdir / f"{stem}.json", outdir / f"{stem}.md"
    j.write_text(json.dumps(p, indent=2), encoding="utf-8")
    m.write_text(to_markdown(p), encoding="utf-8")
    return j, m
