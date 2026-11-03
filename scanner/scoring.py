"""Scanner scoring: percentile ranks, gates, group scores, blend and top-N selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scanner.indicators import latest_metrics

REQUIRED = ["close", "sma200", "ret_63d", "atr14", "sector"]

GROUPS = {
    "technical": ["sma_trend", "rsi", "macd", "volume"],
    "fundamental": ["valuation_pe", "ev_ebitda", "fcf_yield"],
    "sentiment": ["analyst"],
}

_FUND_COLS = ["pe", "ev_ebitda", "fcf_yield", "sector"]
_ANALYST_COLS = ["rec_mean", "target_gap", "rec_trend"]


def percentile_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0-100 percentile rank (lowest 0, highest 100, ties averaged); NaN stays NaN."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if not higher_is_better:
        s = -s
    n = s.notna().sum()
    if n == 0:
        return pd.Series(np.nan, index=s.index)
    if n == 1:
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    return (s.rank(method="average") - 1) / (n - 1) * 100


def _day(ts) -> pd.Timestamp:
    """Tz-naive, midnight-normalised timestamp (NaT passes through)."""
    ts = pd.Timestamp(ts)
    if ts is pd.NaT:
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def build_metric_table(prices: dict[str, pd.DataFrame], fundamentals: pd.DataFrame,
                       analyst: pd.DataFrame, earnings: pd.Series,
                       as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per ticker in `prices`: latest metrics, fundamentals, analyst data,
    `sector`, `last_bar`, `next_earnings` and `earnings_known`. Missing values stay NaN."""
    rows = {}
    for ticker, df in prices.items():
        m = latest_metrics(df)
        m["last_bar"] = _day(df.index[-1]) if len(df) else pd.NaT
        rows[ticker] = m
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "ticker"
    tickers = table.index
    fund = fundamentals.reindex(tickers, columns=_FUND_COLS)
    an = analyst.reindex(tickers, columns=_ANALYST_COLS)
    table = table.join(fund).join(an)
    ne = earnings.reindex(tickers)
    table["next_earnings"] = pd.to_datetime(
        pd.Series([_day(x) for x in ne], index=tickers, dtype=object), errors="coerce")
    table["earnings_known"] = table["next_earnings"].notna()
    return table


def _missing(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def apply_gates(table: pd.DataFrame, as_of: pd.Timestamp, stale_days: int = 5,
                earnings_blackout_days: int = 5) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (candidates, {ticker: reason}). Order: stale_data, missing_data,
    trend, momentum, earnings_blackout. Unknown earnings date does not reject."""
    as_of = _day(as_of)
    blackout_end = pd.bdate_range(as_of + pd.Timedelta(days=1),
                                  periods=earnings_blackout_days)[-1]
    rejected: dict[str, str] = {}
    keep = []
    for ticker, r in table.iterrows():
        last = r.get("last_bar", pd.NaT)
        ne = r.get("next_earnings", pd.NaT)
        ne = pd.NaT if _missing(ne) else _day(ne)
        if _missing(last) or (as_of - _day(last)).days > stale_days:
            reason = "stale_data"
        elif any(_missing(r.get(c)) for c in REQUIRED):
            reason = "missing_data"
        elif not r["close"] > r["sma200"]:
            reason = "trend"
        elif not r["ret_63d"] > 0:
            reason = "momentum"
        elif ne is not pd.NaT and as_of <= ne <= blackout_end:
            reason = "earnings_blackout"
        else:
            keep.append(ticker)
            continue
        rejected[ticker] = reason
    return table.loc[keep], rejected


def _num(table: pd.DataFrame, col: str) -> pd.Series:
    if col not in table.columns:
        return pd.Series(np.nan, index=table.index)
    return pd.to_numeric(table[col], errors="coerce").astype(float)


def _row_mean(ranks: list[pd.Series], index) -> pd.Series:
    if not ranks:
        return pd.Series(np.nan, index=index)
    return pd.concat(ranks, axis=1).mean(axis=1, skipna=True)


def _toggle_rank(table: pd.DataFrame, toggle: str) -> pd.Series:
    if toggle == "sma_trend":
        return percentile_rank(_num(table, "close") / _num(table, "sma50") - 1)
    if toggle == "rsi":
        return percentile_rank(-(_num(table, "rsi14") - 55).abs())
    if toggle == "macd":
        return percentile_rank(_num(table, "macd_hist") / _num(table, "close"))
    if toggle == "volume":
        return percentile_rank(_num(table, "vol_trend"))
    if toggle == "valuation_pe":
        pe = _num(table, "pe")
        return percentile_rank(-pe.where(pe > 0))
    if toggle == "ev_ebitda":
        ev = _num(table, "ev_ebitda")
        return percentile_rank(-ev.where(ev > 0))
    if toggle == "fcf_yield":
        return percentile_rank(_num(table, "fcf_yield"))
    if toggle == "analyst":
        return _row_mean([
            percentile_rank(_num(table, "rec_mean"), higher_is_better=False),
            percentile_rank(_num(table, "target_gap")),
            percentile_rank(_num(table, "rec_trend")),
        ], table.index)
    raise ValueError(f"unknown toggle: {toggle}")


def score_table(table: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add rank_<toggle> for each enabled toggle, then technical, fundamental,
    sentiment, risk and blend. Ranks are across the rows of `table`."""
    out = table.copy()
    toggles = list(settings["toggles"])
    weights = settings["weights"]
    for t in toggles:
        out[f"rank_{t}"] = _toggle_rank(table, t)
    for group, members in GROUPS.items():
        enabled = [t for t in toggles if t in members]
        if enabled:
            out[group] = out[[f"rank_{t}" for t in enabled]].fillna(0).mean(axis=1)
        else:
            out[group] = 0.0
    out["risk"] = _row_mean([
        percentile_rank(_num(table, "vol_20d_ann"), higher_is_better=False),
        percentile_rank(_num(table, "atr_pct"), higher_is_better=False),
        percentile_rank(_num(table, "drawdown")),
    ], table.index).fillna(0)
    out["blend"] = sum(out[g] * float(weights.get(g, 0.0))
                       for g in ("technical", "fundamental", "sentiment", "risk"))
    return out


def _reasons(row: pd.Series, rank_cols: list[str], k: int = 3) -> list[str]:
    vals = [(c[len("rank_"):], row[c]) for c in rank_cols if not _missing(row[c])]
    vals.sort(key=lambda kv: -kv[1])  # stable: ties keep toggle order
    return [f"{name} {round(float(v))}" for name, v in vals[:k]]


def select_top(scored: pd.DataFrame, n: int = 3, max_per_sector: int = 2) -> pd.DataFrame:
    """Top n by blend (desc, ties by ticker), at most max_per_sector per sector.
    Returns fewer than n rows when fewer qualify; adds a `reasons` list column."""
    order = sorted(scored.index, key=lambda t: (-float(scored.loc[t, "blend"]), str(t)))
    per_sector: dict = {}
    chosen = []
    for t in order:
        if len(chosen) >= n:
            break
        sec = scored.loc[t, "sector"]
        if per_sector.get(sec, 0) >= max_per_sector:
            continue
        per_sector[sec] = per_sector.get(sec, 0) + 1
        chosen.append(t)
    top = scored.loc[chosen].copy()
    rank_cols = [c for c in scored.columns if c.startswith("rank_")]
    top["reasons"] = pd.Series([_reasons(top.loc[t], rank_cols) for t in chosen],
                               index=top.index, dtype=object)
    return top
