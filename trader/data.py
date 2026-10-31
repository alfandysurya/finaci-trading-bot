"""Price data: download, cache, load.

Returns a dict {ticker: DataFrame[Open, High, Low, Close, Volume]} with split/dividend
adjusted prices on a DatetimeIndex.

Caveat (survivorship bias): the default universe is *today's* large caps, so any
backtest on it is optimistic — those names are in the list because they did well.
Treat absolute returns with suspicion; compare strategies against buy-and-hold on
the same universe instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Open", "Close"])


def download_yfinance(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf  # imported lazily so tests run without it

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      group_by="ticker", threads=False)  # threaded bulk calls dropped most tickers
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = _clean(df)
        if len(df):
            out[t] = df
    missing = sorted(set(tickers) - set(out))
    if missing:
        print(f"[data] no data for: {', '.join(missing)}")
    return out


def load_csv_dir(csv_dir: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        p = csv_dir / f"{t}.csv"
        if p.exists():
            out[t] = _clean(pd.read_csv(p, index_col=0, parse_dates=True))
    return out


def save_cache(prices: dict[str, pd.DataFrame], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for t, df in prices.items():
        df.to_parquet(cache_dir / f"{t}.parquet")


def load_cache(cache_dir: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        p = cache_dir / f"{t}.parquet"
        if p.exists():
            out[t] = pd.read_parquet(p)
    return out


def _require_all(prices: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, pd.DataFrame]:
    missing = sorted(set(tickers) - set(prices))
    if missing:
        raise RuntimeError(f"No price data for: {', '.join(missing)}. Stopping rather than trading on partial data.")
    return prices


def get_prices(cfg: dict, refresh: bool = False, extra: list[str] | None = None) -> dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys(cfg["universe"]["tickers"] + [cfg["universe"]["benchmark"]] + (extra or [])))
    dcfg = cfg["data"]
    cache_dir = ROOT / dcfg["cache_dir"]

    if dcfg["source"] == "csv":
        return _require_all(load_csv_dir(ROOT / dcfg["csv_dir"], tickers), tickers)

    if not refresh:
        cached = load_cache(cache_dir, tickers)
        if len(cached) == len(tickers):
            return cached

    prices = _require_all(download_yfinance(tickers, dcfg["start"]), tickers)
    save_cache(prices, cache_dir)
    return prices


def synthetic_prices(tickers: list[str], start: str = "2012-01-01", end: str = "2026-10-30",
                     seed: int = 7) -> dict[str, pd.DataFrame]:
    """Regime-switching random walks for testing the pipeline offline.

    Trending and choppy regimes alternate, so trend indicators have *something* to find
    in some periods and nothing in others — useful for checking that walk-forward does
    not blindly trust in-sample winners. Do not read anything into results on this data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    market = np.zeros(n)
    i = 0
    while i < n:  # market regimes of 3-18 months
        length = int(rng.integers(60, 380))
        drift = rng.choice([0.0012, -0.0010, 0.0002], p=[0.45, 0.2, 0.35])
        market[i:i + length] = drift
        i += length
    out = {}
    for t in tickers:
        beta = rng.uniform(0.6, 1.6)
        vol = rng.uniform(0.012, 0.03)
        idio = rng.normal(0, vol, n)
        ret = beta * (market + rng.normal(0, 0.008, n)) + idio
        close = 100 * np.exp(np.cumsum(ret))
        gap = rng.normal(0, vol / 3, n)
        open_ = close / np.exp(ret) * np.exp(gap)  # prior close shifted by an overnight gap
        open_[0] = close[0]
        hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
        lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
        vol_ = rng.integers(1_000_000, 20_000_000, n)
        out[t] = pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close,
                               "Volume": vol_}, index=idx)
    return out
