from datetime import datetime, timezone

import pandas as pd
import pytest

from trader import proposals as prop
from trader import simulate as sim
from trader.config import cost_per_side, load_config
from trader.data import get_prices, synthetic_prices

NOW = datetime(2026, 10, 30, 22, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["universe"]["tickers"] = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    c["universe"]["bias_check_tickers"] = ["GGG", "HHH", "III", "JJJ"]
    return c


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "SPY"],
                            start="2014-01-01", end="2026-10-30")


def test_cost_includes_uncapped_jfx_fee():
    # orders stay below the $200 where the $0.10 JFX/KBI cap starts to bind
    assert cost_per_side(load_config()) == pytest.approx((0.003 + 0.0005) * 1.11 + 0.0005)


def test_simulation_trades_only_after_month_end(prices, cfg):
    res = sim.simulate(prices, cfg["universe"]["tickers"], cfg, 0.004, "2016-01-01")
    days = res.order_dates
    assert len(days) > 0
    # every order fills on the first session of a month
    idx = prices["SPY"].index
    firsts = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("min")
    assert set(days) <= set(firsts)


def test_simulation_charges_costs(prices, cfg):
    tickers = cfg["universe"]["tickers"]
    free = sim.simulate(prices, tickers, cfg, 0.0, "2016-01-01")
    paid = sim.simulate(prices, tickers, cfg, 0.01, "2016-01-01")
    assert (1 + paid.returns).prod() < (1 + free.returns).prod()


def test_gate_requires_beating_spy_on_every_universe():
    win, lose = {"sharpe": 1.0}, {"sharpe": 0.5}
    spy = {"sharpe": 0.8}
    assert sim.go_live({"a": win, "b": win}, spy)
    assert not sim.go_live({"a": win, "b": lose}, spy)


def test_report_is_utf8_and_states_verdict(prices, cfg, tmp_path):
    out = sim.validate(prices, cfg, 0.004, tmp_path)
    text = (tmp_path / "validation_report.md").read_text(encoding="utf-8")
    assert isinstance(out["go_live"], bool)
    assert ("APPROVED" in text) or ("NOT APPROVED" in text)


def test_proposal_flags_unapproved_strategy(prices, cfg):
    sel = cfg["strategy"]
    p = prop.build(prices, sel, cfg, holdings={}, now=NOW, go_live=False)
    assert any("PAPER ONLY" in n for n in p["notes"])


def test_held_ticker_missing_from_data_stops_the_run(prices, cfg):
    partial = {k: v for k, v in prices.items() if k != "AAA"}
    with pytest.raises(ValueError, match="AAA"):
        prop.build(partial, cfg["strategy"], cfg, holdings={"cash_usd": 1, "positions": {"AAA": 1.0}}, now=NOW)


def test_stale_ticker_stops_the_run(prices, cfg):
    stale = dict(prices)
    stale["BBB"] = prices["BBB"].iloc[:-3]
    with pytest.raises(ValueError, match="BBB"):
        prop.build(stale, cfg["strategy"], cfg, holdings={}, now=NOW)


def test_missing_ticker_in_data_load_stops_the_run(prices, tmp_path):
    prices["AAA"].to_csv(tmp_path / "AAA.csv")
    c = {"universe": {"tickers": ["AAA", "BBB"], "benchmark": "AAA"},
         "data": {"source": "csv", "csv_dir": str(tmp_path), "cache_dir": str(tmp_path)}}
    with pytest.raises(RuntimeError, match="BBB"):
        get_prices(c)
