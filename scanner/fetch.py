"""Thin yfinance adapters returning plain DataFrames/Series with fixed schemas."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from trader.data import download_yfinance

FUNDAMENTAL_COLS = ["pe", "ev_ebitda", "fcf_yield", "sector"]
ANALYST_COLS = ["rec_mean", "target_gap", "rec_trend"]
_BUY_COLS = ["strongBuy", "buy"]
_ALL_COLS = ["strongBuy", "buy", "hold", "sell", "strongSell"]


def fetch_prices(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    return download_yfinance(tickers, start)


def _default_factory(ticker_factory):
    if ticker_factory is not None:
        return ticker_factory
    import yfinance as yf  # lazy so tests run without network/yfinance

    return yf.Ticker


def _num(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


def _fundamentals_row(info: dict) -> dict:
    cap, fcf = _num(info.get("marketCap")), _num(info.get("freeCashflow"))
    fcf_yield = fcf / cap if np.isfinite(cap) and cap > 0 and np.isfinite(fcf) else np.nan
    sector = info.get("sector")
    return {
        "pe": _num(info.get("trailingPE")),
        "ev_ebitda": _num(info.get("enterpriseToEbitda")),
        "fcf_yield": fcf_yield,
        "sector": sector if isinstance(sector, str) else None,
    }


def _buy_share(summary: pd.DataFrame, period: str) -> float:
    rows = summary[summary["period"] == period]
    if rows.empty:
        return np.nan
    row = rows.iloc[0]
    total = sum(_num(row.get(c)) for c in _ALL_COLS)
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return sum(_num(row.get(c)) for c in _BUY_COLS) / total


def _analyst_row(tk) -> dict:
    info = tk.info or {}
    price = _num(info.get("currentPrice"))
    if not np.isfinite(price):
        price = _num(info.get("regularMarketPrice"))
    target = _num(info.get("targetMeanPrice"))
    gap = target / price - 1 if np.isfinite(price) and price > 0 and np.isfinite(target) else np.nan
    trend = np.nan
    summary = tk.recommendations_summary
    if isinstance(summary, pd.DataFrame) and "period" in summary.columns:
        trend = _buy_share(summary, "0m") - _buy_share(summary, "-3m")
    return {"rec_mean": _num(info.get("recommendationMean")), "target_gap": gap,
            "rec_trend": trend}


def _next_earnings(tk) -> pd.Timestamp:
    cal = tk.calendar
    dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
    if dates is None:
        return pd.NaT
    if not isinstance(dates, (list, tuple, pd.Series, np.ndarray)):
        dates = [dates]
    today = pd.Timestamp.now().normalize()
    parsed = []
    for d in dates:
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            continue
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        parsed.append(ts)
    future = [d for d in parsed if d.normalize() >= today]
    return min(future) if future else pd.NaT


def _collect(tickers, ticker_factory, row_fn, empty_row, label):
    make = _default_factory(ticker_factory)
    rows = {}
    for t in tickers:
        try:
            rows[t] = row_fn(make(t))
        except Exception as exc:  # one bad ticker must not abort the batch
            print(f"[fetch] {label} failed for {t}: {type(exc).__name__}: {exc}")
            rows[t] = empty_row
    return rows


def fetch_fundamentals(tickers: list[str],
                       ticker_factory: Callable[[str], Any] | None = None) -> pd.DataFrame:
    empty = {"pe": np.nan, "ev_ebitda": np.nan, "fcf_yield": np.nan, "sector": None}
    rows = _collect(tickers, ticker_factory, lambda tk: _fundamentals_row(tk.info or {}),
                    empty, "fundamentals")
    df = pd.DataFrame.from_dict(rows, orient="index", columns=FUNDAMENTAL_COLS)
    return df.reindex(list(tickers))


def fetch_analyst(tickers: list[str],
                  ticker_factory: Callable[[str], Any] | None = None) -> pd.DataFrame:
    empty = {c: np.nan for c in ANALYST_COLS}
    rows = _collect(tickers, ticker_factory, _analyst_row, empty, "analyst")
    df = pd.DataFrame.from_dict(rows, orient="index", columns=ANALYST_COLS).astype(float)
    return df.reindex(list(tickers))


def fetch_earnings(tickers: list[str],
                   ticker_factory: Callable[[str], Any] | None = None) -> pd.Series:
    rows = _collect(tickers, ticker_factory, _next_earnings, pd.NaT, "earnings")
    s = pd.Series({t: rows[t] for t in tickers}, dtype="datetime64[ns]")
    return s.reindex(list(tickers))
