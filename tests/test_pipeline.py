import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trader import backtest as bt
from trader import proposals as prop
from trader import walkforward as wfm
from trader.config import cost_per_side, load_config
from trader.data import synthetic_prices
from trader.indicators import all_configs, target_position, rsi


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["universe"]["tickers"] = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    c["universe"]["benchmark"] = "SPY"
    return c


@pytest.fixture(scope="module")
def prices(cfg):
    return synthetic_prices(cfg["universe"]["tickers"] + ["SPY"], start="2014-01-01", end="2026-10-30")


def test_cost_matches_pluang_schedule(cfg):
    # (0.30% fee + 0.05% JFX/KBI) * 1.11 VAT + 5 bps slippage
    assert cost_per_side(cfg) == pytest.approx(0.0035 * 1.11 + 0.0005)


def test_no_lookahead(prices):
    """Changing the future must not change today's signal."""
    df = prices["AAA"]
    for name, p in all_configs():
        full = target_position(df, name, p)
        cut = target_position(df.iloc[:2000], name, p)
        pd.testing.assert_series_equal(full.iloc[:2000], cut, check_names=False)


def test_signals_are_binary(prices):
    for name, p in all_configs():
        s = target_position(prices["BBB"], name, p)
        assert set(s.unique()) <= {0.0, 1.0}


def test_rsi_bounds(prices):
    r = rsi(prices["CCC"].Close, 14).dropna()
    assert r.between(0, 100).all()


def test_backtest_timing_and_costs():
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"Open": [100, 101, 110, 120, 130], "Close": [100, 105, 115, 125, 128],
                       "High": 0, "Low": 0, "Volume": 0}, index=idx, dtype=float)
    tgt = pd.Series([1, 1, 0, 0, 0], index=idx, dtype=float)   # signal on day 0 close, exit signal day 2
    res = bt.strategy_returns(df, tgt, cost=0.01)
    # day1: enter at open 101 -> close 105, minus cost
    assert res.ret.iloc[1] == pytest.approx(105 / 101 - 1 - 0.01)
    # day2: hold 105 -> 115
    assert res.ret.iloc[2] == pytest.approx(115 / 105 - 1)
    # day3: exit at open 120 from close 115, minus cost
    assert res.ret.iloc[3] == pytest.approx(120 / 115 - 1 - 0.01)
    assert res.ret.iloc[4] == 0


def test_walkforward_runs(prices, cfg):
    res = wfm.run_walkforward(prices, cfg, cost_per_side(cfg))
    assert len(res.folds) >= 5
    assert {"selected", "buy_hold", "baseline_sma50_200"} <= set(res.oos.columns)
    # out-of-sample periods are contiguous calendar years after the first train window
    assert res.oos.index.is_monotonic_increasing
    assert res.final_selection["name"] in {n for n, _ in all_configs()}


def test_proposals_respect_limits(prices, cfg):
    sel = {"name": "price_above_sma", "params": {"n": 50}}
    now = datetime(2026, 10, 30, 22, tzinfo=timezone.utc)
    p = prop.build(prices, sel, cfg, holdings={}, now=now)
    tw = [t["target_weight"] for t in p["targets"]]
    assert all(w <= cfg["portfolio"]["max_weight"] + 1e-9 for w in tw)
    assert sum(tw) <= 1 - cfg["portfolio"]["cash_buffer"] + 1e-6
    assert len([w for w in tw if w > 0]) <= cfg["portfolio"]["max_positions"]
    buys = sum(o["usd_amount"] for o in p["orders"] if o["side"] == "BUY")
    assert buys <= p["limits"]["max_total_buy_usd"] + 0.01
    json.dumps(p)  # serialisable


def test_proposals_exit_positions_without_signal(prices, cfg):
    sel = {"name": "price_above_sma", "params": {"n": 50}}
    flat = [t for t in cfg["universe"]["tickers"]
            if target_position(prices[t], "price_above_sma", {"n": 50}).iloc[-1] == 0]
    if not flat:
        pytest.skip("every synthetic ticker is long today")
    holdings = {"cash_usd": 100, "positions": {flat[0]: 2.0}}
    p = prop.build(prices, sel, cfg, holdings=holdings,
                   now=datetime(2026, 10, 30, 22, tzinfo=timezone.utc))
    sells = [o for o in p["orders"] if o["ticker"] == flat[0]]
    assert sells and sells[0]["side"] == "SELL" and sells[0]["sell_all"]
    assert p["orders"][0]["side"] == "SELL"  # sells go first
