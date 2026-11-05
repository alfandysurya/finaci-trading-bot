import math

import numpy as np
import pandas as pd
import pytest

from scanner.config import active_settings, load_scanner_config
from scanner.scoring import (
    apply_gates,
    build_metric_table,
    percentile_rank,
    score_table,
    select_top,
)

AS_OF = pd.Timestamp("2026-10-30")  # a Friday


def _row(**over):
    base = {
        "close": 100.0, "sma50": 95.0, "sma200": 90.0, "rsi14": 55.0,
        "macd_hist": 1.0, "vol_trend": 1.1, "ret_63d": 0.05, "atr14": 2.0,
        "atr_pct": 0.02, "drawdown": -0.05, "vol_20d_ann": 0.25,
        "dollar_vol_20d": 1e8, "pe": 20.0, "ev_ebitda": 12.0, "fcf_yield": 0.04,
        "rec_mean": 2.0, "target_gap": 0.1, "rec_trend": 0.1, "sector": "Tech",
        "last_bar": AS_OF, "next_earnings": pd.NaT, "earnings_known": False,
    }
    base.update(over)
    return base


def _table(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


def _settings(toggles=None):
    cfg = load_scanner_config()
    s = {"profile": "balanced"}
    if toggles is not None:
        s["toggles"] = toggles
    return active_settings(cfg, s)


def test_percentile_rank_ordering_and_nan():
    s = pd.Series({"a": 1.0, "b": np.nan, "c": 3.0, "d": 2.0})
    r = percentile_rank(s)
    assert r["a"] == 0.0 and r["d"] == 50.0 and r["c"] == 100.0
    assert math.isnan(r["b"])
    lo = percentile_rank(s, higher_is_better=False)
    assert lo["a"] == 100.0 and lo["c"] == 0.0
    assert percentile_rank(pd.Series({"x": 5.0})).tolist() == [50.0]


def test_gate_rejects_below_sma200():
    t = _table({"OK": _row(), "BAD": _row(close=80.0)})
    cands, rejected = apply_gates(t, AS_OF)
    assert list(cands.index) == ["OK"]
    assert rejected == {"BAD": "trend"}


def test_gate_rejects_negative_63d_return():
    t = _table({"OK": _row(), "BAD": _row(ret_63d=-0.01)})
    cands, rejected = apply_gates(t, AS_OF)
    assert list(cands.index) == ["OK"]
    assert rejected == {"BAD": "momentum"}


def test_gate_earnings_blackout_five_trading_days():
    days = pd.bdate_range(AS_OF + pd.Timedelta(days=1), periods=6)
    t = _table({
        "D4": _row(next_earnings=days[3], earnings_known=True),
        "D5": _row(next_earnings=days[4], earnings_known=True),
        "D6": _row(next_earnings=days[5], earnings_known=True),
        "TODAY": _row(next_earnings=AS_OF, earnings_known=True),
        "PAST": _row(next_earnings=AS_OF - pd.Timedelta(days=3), earnings_known=True),
    })
    cands, rejected = apply_gates(t, AS_OF)
    assert rejected == {"D4": "earnings_blackout", "D5": "earnings_blackout",
                        "TODAY": "earnings_blackout"}
    assert sorted(cands.index) == ["D6", "PAST"]


def test_gate_missing_required_metric_reported_not_guessed():
    t = _table({
        "OK": _row(),
        "NOSMA": _row(sma200=np.nan),
        "NOSEC": _row(sector=None),
        "NOATR": _row(atr14=np.nan),
    })
    cands, rejected = apply_gates(t, AS_OF)
    assert list(cands.index) == ["OK"]
    assert rejected == {"NOSMA": "missing_data", "NOSEC": "missing_data",
                        "NOATR": "missing_data"}


def test_gate_rejects_stale_data():
    t = _table({
        "OK": _row(last_bar=AS_OF - pd.Timedelta(days=5)),
        "OLD": _row(last_bar=AS_OF - pd.Timedelta(days=6), sma200=np.nan),
    })
    cands, rejected = apply_gates(t, AS_OF)
    assert list(cands.index) == ["OK"]
    assert rejected == {"OLD": "stale_data"}  # stale checked before missing


def test_unknown_earnings_not_rejected():
    idx = pd.bdate_range(end=AS_OF, periods=260)
    close = pd.Series(np.linspace(50, 100, len(idx)), index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                       "Close": close, "Volume": 1e6}, index=idx)
    fund = pd.DataFrame({"pe": [20.0], "ev_ebitda": [10.0], "fcf_yield": [0.03],
                         "sector": ["Tech"]}, index=["AAA"])
    analyst = pd.DataFrame({"rec_mean": [2.0], "target_gap": [0.1], "rec_trend": [0.0]},
                           index=["AAA"])
    earnings = pd.Series({"AAA": pd.NaT})
    table = build_metric_table({"AAA": df}, fund, analyst, earnings, AS_OF)
    assert bool(table.loc["AAA", "earnings_known"]) is False
    assert table.loc["AAA", "sector"] == "Tech"
    assert table.loc["AAA", "last_bar"] == AS_OF
    cands, rejected = apply_gates(table, AS_OF)
    assert rejected == {}
    assert list(cands.index) == ["AAA"]


def _three():
    # Every metric distinct across the three, so each rank is 0, 50 or 100.
    return _table({
        "A": _row(sma50=90.0, rsi14=55.0, macd_hist=1.0, vol_trend=1.2,
                  pe=10.0, ev_ebitda=8.0, fcf_yield=0.02,
                  rec_mean=1.5, target_gap=0.10, rec_trend=0.3,
                  vol_20d_ann=0.2, atr_pct=0.01, drawdown=-0.05),
        "B": _row(sma50=95.0, rsi14=70.0, macd_hist=2.0, vol_trend=0.8,
                  pe=20.0, ev_ebitda=12.0, fcf_yield=0.05,
                  rec_mean=2.5, target_gap=0.20, rec_trend=0.1,
                  vol_20d_ann=0.3, atr_pct=0.02, drawdown=-0.01, sector="Health"),
        "C": _row(sma50=100.0, rsi14=50.0, macd_hist=0.5, vol_trend=1.0,
                  pe=30.0, ev_ebitda=15.0, fcf_yield=0.03,
                  rec_mean=2.0, target_gap=0.05, rec_trend=-0.1,
                  vol_20d_ann=0.4, atr_pct=0.03, drawdown=-0.20, sector="Energy"),
    })


def test_disabled_toggle_does_not_affect_score():
    toggles = ["sma_trend", "macd", "volume", "valuation_pe", "ev_ebitda",
               "fcf_yield", "analyst"]
    scored = score_table(_three(), _settings(toggles))
    assert "rank_rsi" not in scored.columns
    expected = scored[["rank_sma_trend", "rank_macd", "rank_volume"]].mean(axis=1)
    pd.testing.assert_series_equal(scored["technical"], expected, check_names=False)
    assert scored.loc["A", "technical"] == pytest.approx((100 + 50 + 100) / 3)


def test_blend_uses_profile_weights():
    legacy = ["sma_trend", "rsi", "macd", "volume", "valuation_pe", "ev_ebitda",
              "fcf_yield", "analyst"]
    scored = score_table(_three(), _settings(legacy))
    # Hand-computed ranks:
    # A: tech (100,100,50,100)=87.5; fund (100,100,0)=66.667;
    #    sent mean(100,50,100)=83.333; risk mean(100,100,50)=83.333
    #    blend = .3*87.5 + .3*66.667 + .2*83.333 + .2*83.333 = 79.5833
    # B: tech (50,0,100,0)=37.5; fund (50,50,100)=66.667; sent (0,100,50)=50;
    #    risk (50,50,100)=66.667 -> 11.25 + 20 + 10 + 13.333 = 54.5833
    # C: tech (0,50,0,50)=25; fund (0,0,50)=16.667; sent (50,0,0)=16.667;
    #    risk (0,0,0)=0 -> 7.5 + 5 + 3.333 + 0 = 15.8333
    assert scored.loc["A", "technical"] == pytest.approx(87.5)
    assert scored.loc["B", "sentiment"] == pytest.approx(50.0)
    assert scored.loc["C", "risk"] == pytest.approx(0.0)
    assert scored.loc["A", "blend"] == pytest.approx(79.583333, abs=1e-4)
    assert scored.loc["B", "blend"] == pytest.approx(54.583333, abs=1e-4)
    assert scored.loc["C", "blend"] == pytest.approx(15.833333, abs=1e-4)
    top = select_top(scored)
    assert list(top.index) == ["A", "B", "C"]
    assert top.loc["A", "reasons"] == ["sma_trend 100", "rsi 100", "volume 100"]


def test_select_top_sector_cap():
    scored = _table({
        "T1": {"sector": "Tech", "blend": 90.0, "rank_sma_trend": 91.0},
        "T2": {"sector": "Tech", "blend": 85.0, "rank_sma_trend": 80.0},
        "T3": {"sector": "Tech", "blend": 80.0, "rank_sma_trend": 70.0},
        "T4": {"sector": "Tech", "blend": 75.0, "rank_sma_trend": 60.0},
        "H1": {"sector": "Health", "blend": 70.0, "rank_sma_trend": np.nan},
    })
    top = select_top(scored, n=3, max_per_sector=2)
    assert list(top.index) == ["T1", "T2", "H1"]
    assert top.loc["T1", "reasons"] == ["sma_trend 91"]
    assert top.loc["H1", "reasons"] == []


def test_select_top_returns_fewer_than_n_when_few_pass():
    scored = score_table(_three().loc[["A", "B"]], _settings())
    top = select_top(scored, n=3)
    assert len(top) == 2
    assert list(top.index) == ["A", "B"]
    empty = select_top(scored.iloc[0:0], n=3)
    assert len(empty) == 0 and "reasons" in empty.columns


def test_select_top_ties_broken_by_ticker():
    scored = _table({
        "ZZZ": {"sector": "Tech", "blend": 50.0},
        "AAA": {"sector": "Energy", "blend": 50.0},
    })
    assert list(select_top(scored).index) == ["AAA", "ZZZ"]


def test_guide_toggles_rank_direction_and_der_exemption():
    import numpy as np
    import pandas as pd
    from scanner.scoring import _toggle_rank
    t = pd.DataFrame({"peg": [0.5, 2.0, -1.0], "forward_pe": [10.0, 30.0, np.nan],
                      "roe": [0.30, 0.05, 0.1], "eps_growth": [0.4, -0.2, 0.0],
                      "debt_equity": [0.2, 3.0, 9.0],
                      "sector": ["Tech", "Tech", "Financials"]},
                     index=["A", "B", "C"])
    for tog in ("peg", "forward_pe", "roe", "eps_growth", "debt_equity"):
        r = _toggle_rank(t, tog)
        assert r["A"] > r["B"], tog
    assert np.isnan(_toggle_rank(t, "peg")["C"])      # negative PEG is not ranked
    assert _toggle_rank(t, "debt_equity")["C"] == 50  # bank: DER ignored, neutral
