"""Thin yfinance adapters returning plain DataFrames/Series with fixed schemas.

The per-ticker fetches (fundamentals, analyst, earnings) are throttled: a pause
between tickers, retries with growing waits when Yahoo answers "Too Many Requests",
and a circuit breaker that stops calling Yahoo for the rest of a fetch after
BREAKER_AFTER consecutive tickers stayed rate limited (their rows are left empty).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from trader.data import download_yfinance

FUNDAMENTAL_COLS = ["pe", "ev_ebitda", "fcf_yield", "sector"]
ANALYST_COLS = ["rec_mean", "target_gap", "rec_trend"]
_BUY_COLS = ["strongBuy", "buy"]
_ALL_COLS = ["strongBuy", "buy", "hold", "sell", "strongSell"]

MIN_INTERVAL = 0.25          # seconds between tickers
RETRY_WAITS = (5, 15, 45)    # seconds before each retry of a rate-limited ticker
BREAKER_AFTER = 8            # consecutive rate-limited tickers before giving up the fetch

RETRY_BUDGET = 480           # seconds of retry waits allowed per run, across all fetches

# Worst-case wall time: with a shared RateLimitState, all retry waits (5/15/45 s) in one
# run add up to at most RETRY_BUDGET = 480 s (8 min), whatever the refusal pattern
# (steady or on-and-off). After that, and after any breaker trip, every ticker gets a
# single attempt, so the rest costs only the 0.25 s pauses plus Yahoo's own response
# time. A fully refused run therefore spends at most about 8 min in retry waits in
# total, under the 15-minute bound and the 30-minute workflow timeout (the weekly
# fetches earnings first and aborts right after it when most dates are unknown).


@dataclass
class RateLimitState:
    """Shared by the fetches of one run. Once any breaker trips, or the retry waits
    would exceed `retry_budget` seconds in total, every later attempt is a single one
    (no retry waits) for the rest of the run."""

    tripped: bool = False
    retry_budget: float = RETRY_BUDGET
    waited: float = 0.0  # seconds slept in retry waits so far (not the min interval)

    def allow_wait(self, seconds: float) -> bool:
        """Book a retry wait if it fits the budget; otherwise trip and refuse."""
        if self.tripped or self.waited + seconds > self.retry_budget:
            self.tripped = True
            return False
        self.waited += seconds
        return True


def _sleep(seconds: float) -> None:  # module-level so tests can replace it
    time.sleep(seconds)


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


def _is_rate_limit(exc: BaseException) -> bool:
    return "RateLimit" in type(exc).__name__ or "Too Many Requests" in str(exc)


def _collect(tickers, ticker_factory, row_fn, empty_row, label, *, sleep=None,
             min_interval=MIN_INTERVAL, retry_waits=RETRY_WAITS, breaker_after=BREAKER_AFTER,
             rate_state=None):
    make = _default_factory(ticker_factory)
    sleep = _sleep if sleep is None else sleep
    tickers = list(tickers)
    rows = {}
    limited = 0  # consecutive tickers that ended rate limited
    for i, t in enumerate(tickers):
        if limited >= breaker_after:
            rest = tickers[i:]
            print(f"[fetch] rate limited: skipping {len(rest)} remaining {label}")
            for r in rest:
                rows[r] = empty_row
            break
        if i and min_interval > 0:
            sleep(min_interval)
        rows[t] = empty_row
        for attempt in range(len(retry_waits) + 1):
            try:
                rows[t] = row_fn(make(t))
                limited = 0
                break
            except Exception as exc:  # one bad ticker must not abort the batch
                rate_limited = _is_rate_limit(exc)
                if (rate_limited and attempt < len(retry_waits)
                        and (rate_state is None or rate_state.allow_wait(retry_waits[attempt]))):
                    sleep(retry_waits[attempt])
                    continue
                print(f"[fetch] {label} failed for {t}: {type(exc).__name__}: {exc}")
                limited = limited + 1 if rate_limited else 0
                if limited >= breaker_after and rate_state is not None:
                    rate_state.tripped = True  # even when this was the last ticker
                break
    return rows


def _throttle(sleep, min_interval, retry_waits, breaker_after, rate_state) -> dict:
    return {"sleep": sleep, "min_interval": min_interval, "retry_waits": tuple(retry_waits),
            "breaker_after": breaker_after, "rate_state": rate_state}


def fetch_fundamentals(tickers: list[str],
                       ticker_factory: Callable[[str], Any] | None = None,
                       *, sleep: Callable[[float], None] | None = None,
                       min_interval: float = MIN_INTERVAL,
                       retry_waits: Sequence[float] = RETRY_WAITS,
                       breaker_after: int = BREAKER_AFTER,
                       rate_state: RateLimitState | None = None) -> pd.DataFrame:
    empty = {"pe": np.nan, "ev_ebitda": np.nan, "fcf_yield": np.nan, "sector": None}
    rows = _collect(tickers, ticker_factory, lambda tk: _fundamentals_row(tk.info or {}),
                    empty, "fundamentals", **_throttle(sleep, min_interval, retry_waits,
                                                       breaker_after, rate_state))
    df = pd.DataFrame.from_dict(rows, orient="index", columns=FUNDAMENTAL_COLS)
    return df.reindex(list(tickers))


def fetch_analyst(tickers: list[str],
                  ticker_factory: Callable[[str], Any] | None = None,
                  *, sleep: Callable[[float], None] | None = None,
                  min_interval: float = MIN_INTERVAL,
                  retry_waits: Sequence[float] = RETRY_WAITS,
                  breaker_after: int = BREAKER_AFTER,
                  rate_state: RateLimitState | None = None) -> pd.DataFrame:
    empty = {c: np.nan for c in ANALYST_COLS}
    rows = _collect(tickers, ticker_factory, _analyst_row, empty, "analyst",
                    **_throttle(sleep, min_interval, retry_waits, breaker_after,
                                rate_state))
    df = pd.DataFrame.from_dict(rows, orient="index", columns=ANALYST_COLS).astype(float)
    return df.reindex(list(tickers))


def fetch_earnings(tickers: list[str],
                   ticker_factory: Callable[[str], Any] | None = None,
                   *, sleep: Callable[[float], None] | None = None,
                   min_interval: float = MIN_INTERVAL,
                   retry_waits: Sequence[float] = RETRY_WAITS,
                   breaker_after: int = BREAKER_AFTER,
                   rate_state: RateLimitState | None = None) -> pd.Series:
    rows = _collect(tickers, ticker_factory, _next_earnings, pd.NaT, "earnings",
                    **_throttle(sleep, min_interval, retry_waits, breaker_after,
                                rate_state))
    s = pd.Series({t: rows[t] for t in tickers}, dtype="datetime64[ns]")
    return s.reindex(list(tickers))
