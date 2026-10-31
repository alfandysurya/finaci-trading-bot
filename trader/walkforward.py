"""Walk-forward indicator selection.

Treats "which indicator?" like model selection with out-of-time validation:
  for each fold: rank every indicator config on the TRAIN window (pooled over the
  universe), pick the top one, then measure it on the following TEST year it never saw.
Stitching the test years gives an honest out-of-sample track record of the *selection
process*, which is what you would actually be trading.

Two diagnostics tell you whether to trust the process at all:
  * rank_corr  — Spearman correlation between train-rank and test-rank of all configs.
                 Near zero = in-sample ranking does not predict the future (selection is noise).
  * selection stability — does the winner keep changing fold to fold?
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import backtest as bt
from .indicators import all_configs, config_id, target_position

BASELINE = ("sma_cross", {"fast": 50, "slow": 200})


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos: pd.DataFrame                    # daily returns: selected, baseline, buy_hold
    oos_metrics: pd.DataFrame
    in_sample: pd.DataFrame              # every config over full history (NOT a recommendation)
    final_selection: dict
    notes: list[str] = field(default_factory=list)


def _panel(prices: dict[str, pd.DataFrame], tickers: list[str], name: str, params: dict,
           cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rets, trades = {}, {}
    for t in tickers:
        df = prices[t]
        res = bt.strategy_returns(df, target_position(df, name, params), cost)
        rets[t], trades[t] = res["ret"], res["trade"]
    return pd.DataFrame(rets), pd.DataFrame(trades)


def _score_window(rets: pd.DataFrame, trades: pd.DataFrame, avail: pd.DataFrame,
                  start, end, score: str) -> tuple[float, float]:
    w = (rets.index >= start) & (rets.index < end)
    r, tr, av = rets[w], trades[w], avail[w]
    sharpes, calmars, tpy = [], [], []
    for t in r.columns:
        rr = r[t][av[t]]
        if len(rr) < 120:
            continue
        m = bt.metrics(rr, trades=tr[t][av[t]])
        sharpes.append(m["sharpe"])
        calmars.append(m["calmar"] if np.isfinite(m["calmar"]) else 0.0)
        tpy.append(m["trades_per_year"])
    if not sharpes:
        return np.nan, np.nan
    s = {"median_sharpe": np.median(sharpes), "mean_sharpe": np.mean(sharpes),
         "median_calmar": np.median(calmars)}[score]
    return float(s), float(np.median(tpy))


def _ew(rets: pd.DataFrame, avail: pd.DataFrame) -> pd.Series:
    """Equal-weight sleeves across tickers that have data that day."""
    r = rets.where(avail)
    return r.mean(axis=1).fillna(0.0)


def run_walkforward(prices: dict[str, pd.DataFrame], cfg: dict, cost: float) -> WalkForwardResult:
    wf = cfg["walkforward"]
    tickers = [t for t in cfg["universe"]["tickers"] if t in prices]
    idx = sorted(set().union(*[prices[t].index for t in tickers]))
    idx = pd.DatetimeIndex(idx)
    prices = {t: prices[t].reindex(idx) for t in tickers}
    # a ticker counts on a day only once it has a full year of history (indicator warm-up)
    avail = pd.DataFrame({t: prices[t].Close.notna() & (prices[t].Close.notna().cumsum() > 252)
                          for t in tickers})
    for t in tickers:  # forward-fill holes so return math works; avail masks them out
        prices[t] = prices[t].ffill()

    configs = all_configs()
    panels = {config_id(n, p): (n, p, *_panel(prices, tickers, n, p, cost)) for n, p in configs}

    bh = pd.DataFrame({t: prices[t].Close.pct_change().fillna(0.0) for t in tickers})

    first_year = (avail.any(axis=1).idxmax()).year + 1   # first full calendar year with data
    last_year = idx[-1].year
    test_years = range(first_year + wf["train_years"], last_year + 1, wf["test_years"])

    fold_rows, oos_parts, notes = [], [], []
    base_id = config_id(*BASELINE)

    for y in test_years:
        tr_start = pd.Timestamp(y - wf["train_years"], 1, 1)
        te_start = pd.Timestamp(y, 1, 1)
        te_end = pd.Timestamp(y + wf["test_years"], 1, 1)

        rows = []
        for cid, (n, p, r, tr) in panels.items():
            s_tr, tpy = _score_window(r, tr, avail, tr_start, te_start, wf["score"])
            s_te, _ = _score_window(r, tr, avail, te_start, te_end, wf["score"])
            rows.append({"config": cid, "train": s_tr, "test": s_te, "tpy": tpy})
        table = pd.DataFrame(rows).dropna(subset=["train"])
        eligible = table[(table.tpy >= wf["min_trades_per_year"]) & (table.tpy <= wf["max_trades_per_year"])]
        if eligible.empty:
            continue
        best = eligible.sort_values("train", ascending=False).iloc[0]
        both = table.dropna(subset=["test"])
        rho = spearmanr(both.train, both.test).statistic if len(both) > 3 else np.nan

        _, _, r_sel, _ = panels[best.config]
        _, _, r_base, _ = panels[base_id]
        w = (idx >= te_start) & (idx < te_end)
        part = pd.DataFrame({
            "selected": _ew(r_sel[w], avail[w]),
            "baseline_sma50_200": _ew(r_base[w], avail[w]),
            "buy_hold": _ew(bh[w], avail[w]),
        })
        oos_parts.append(part)
        fold_rows.append({
            "test_year": y, "selected": best.config, "train_score": best.train,
            "test_score": best.test, "test_rank": int((both.test > best.test).sum()) + 1 if pd.notna(best.test) else None,
            "n_configs": len(table), "rank_corr": rho,
            "oos_return_selected": float((1 + part.selected).prod() - 1),
            "oos_return_buy_hold": float((1 + part.buy_hold).prod() - 1),
        })

    folds = pd.DataFrame(fold_rows)
    oos = pd.concat(oos_parts) if oos_parts else pd.DataFrame()
    oos_metrics = pd.DataFrame({c: bt.metrics(oos[c]) for c in oos.columns}).T if len(oos) else pd.DataFrame()

    # full-history table — shown for transparency, flagged as in-sample
    ins = []
    full_start, full_end = idx[0], idx[-1] + pd.Timedelta(days=1)
    for cid, (n, p, r, tr) in panels.items():
        s, tpy = _score_window(r, tr, avail, full_start, full_end, wf["score"])
        m = bt.metrics(_ew(r, avail))
        ins.append({"config": cid, "family": n, "score": s, "trades_per_year": tpy,
                    "ew_cagr": m["cagr"], "ew_sharpe": m["sharpe"], "ew_max_dd": m["max_drawdown"]})
    in_sample = pd.DataFrame(ins).sort_values("score", ascending=False).reset_index(drop=True)

    # final pick for live use: rank on the most recent train window
    live_start = pd.Timestamp(idx[-1].year - wf["train_years"], idx[-1].month, idx[-1].day)
    rows = []
    for cid, (n, p, r, tr) in panels.items():
        s, tpy = _score_window(r, tr, avail, live_start, full_end, wf["score"])
        rows.append({"config": cid, "name": n, "params": p, "score": s, "tpy": tpy})
    live = pd.DataFrame(rows).dropna(subset=["score"])
    live = live[(live.tpy >= wf["min_trades_per_year"]) & (live.tpy <= wf["max_trades_per_year"])]
    pick = live.sort_values("score", ascending=False).iloc[0]

    # diagnostics -> plain-language notes
    if len(folds):
        mean_rho = folds.rank_corr.mean()
        n_unique = folds.selected.nunique()
        sel_m, bh_m = oos_metrics.loc["selected"], oos_metrics.loc["buy_hold"]
        notes.append(f"Average train->test rank correlation: {mean_rho:.2f} "
                     + ("(in-sample ranking carries some information)." if mean_rho > 0.2 else
                        "(weak: in-sample winners are close to a coin flip out of sample)."))
        notes.append(f"Winner changed across folds: {n_unique} different configs in {len(folds)} folds"
                     + (" — unstable, treat any single pick with caution." if n_unique > len(folds) / 2 else "."))
        notes.append(f"Out-of-sample selected: Sharpe {sel_m.sharpe:.2f}, max DD {sel_m.max_drawdown:.0%}; "
                     f"buy & hold: Sharpe {bh_m.sharpe:.2f}, max DD {bh_m.max_drawdown:.0%}.")
        if sel_m.sharpe < bh_m.sharpe:
            notes.append("The selection process did NOT beat buy & hold on a risk-adjusted basis out of sample. "
                         "Its main use may be drawdown control rather than extra return.")

    final_selection = {
        "name": pick["name"], "params": pick["params"], "config": pick["config"],
        "train_score": float(pick.score), "trades_per_year": float(pick.tpy),
        "train_window": [str(live_start.date()), str(idx[-1].date())],
        "selected_on": str(pd.Timestamp.today().date()),
    }
    return WalkForwardResult(folds, oos, oos_metrics, in_sample, final_selection, notes)


def save(result: WalkForwardResult, outdir) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    result.folds.to_csv(outdir / "wf_folds.csv", index=False)
    result.oos.to_csv(outdir / "wf_oos_daily_returns.csv")
    result.oos_metrics.to_csv(outdir / "wf_oos_metrics.csv")
    result.in_sample.to_csv(outdir / "in_sample_all_configs.csv", index=False)
    with open(outdir / "selection.json", "w") as f:
        json.dump(result.final_selection, f, indent=2)

    pct = lambda x: f"{x:.1%}"
    lines = ["# Walk-forward indicator selection", "",
             f"Live pick: **{result.final_selection['config']}** "
             f"(ranked on {result.final_selection['train_window'][0]} → {result.final_selection['train_window'][1]})", "",
             "## What the diagnostics say", ""] + [f"- {n}" for n in result.notes] + [
             "", "## Out-of-sample (stitched test years, equal-weight across universe, after Pluang costs)", "",
             "| strategy | CAGR | Sharpe | max DD | vol |", "|---|---|---|---|---|"]
    for name, m in result.oos_metrics.iterrows():
        lines.append(f"| {name} | {pct(m.cagr)} | {m.sharpe:.2f} | {pct(m.max_drawdown)} | {pct(m.vol)} |")
    lines += ["", "## Folds", "", "| test year | selected | train score | test score | test rank | rank corr | OOS ret | B&H ret |",
              "|---|---|---|---|---|---|---|---|"]
    for _, f in result.folds.iterrows():
        lines.append(f"| {f.test_year} | {f.selected} | {f.train_score:.2f} | {f.test_score:.2f} | "
                     f"{f.test_rank}/{f.n_configs} | {f.rank_corr:.2f} | {pct(f.oos_return_selected)} | {pct(f.oos_return_buy_hold)} |")
    lines += ["", "## In-sample top 10 (full history — biased upward, NOT the recommendation)", "",
              "| config | score | trades/yr | EW CAGR | EW max DD |", "|---|---|---|---|---|"]
    for _, r in result.in_sample.head(10).iterrows():
        lines.append(f"| {r.config} | {r.score:.2f} | {r.trades_per_year:.1f} | {pct(r.ew_cagr)} | {pct(r.ew_max_dd)} |")
    lines += ["", "_Caveats: universe is today's large caps (survivorship bias); config switches at fold "
              "boundaries are not charged; past performance does not predict future results._"]
    (outdir / "walkforward_report.md").write_text("\n".join(lines), encoding="utf-8")
