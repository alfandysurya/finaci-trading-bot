import numpy as np
import pandas as pd

from scanner.fetch import fetch_analyst, fetch_earnings, fetch_fundamentals


class FakeTicker:
    def __init__(self, info=None, summary=None, calendar=None, boom=False):
        self._info, self._summary, self._cal, self._boom = info, summary, calendar, boom

    @property
    def info(self):
        if self._boom:
            raise RuntimeError("boom")
        return self._info or {}

    @property
    def recommendations_summary(self):
        if self._boom:
            raise RuntimeError("boom")
        return self._summary

    @property
    def calendar(self):
        if self._boom:
            raise RuntimeError("boom")
        return self._cal or {}


def factory(mapping):
    return lambda t: mapping[t]


def test_fundamentals_missing_field_is_nan_not_error():
    f = factory({"AAA": FakeTicker({"trailingPE": 12.0, "marketCap": 100.0,
                                     "freeCashflow": 5.0, "sector": "Tech"})})
    df = fetch_fundamentals(["AAA"], ticker_factory=f)
    assert list(df.columns) == ["pe", "ev_ebitda", "fcf_yield", "sector"]
    assert df.loc["AAA", "pe"] == 12.0
    assert np.isnan(df.loc["AAA", "ev_ebitda"])
    assert df.loc["AAA", "fcf_yield"] == 0.05
    assert df.loc["AAA", "sector"] == "Tech"


def test_fundamentals_fcf_yield_nan_when_market_cap_missing():
    f = factory({"AAA": FakeTicker({"freeCashflow": 5.0}),
                 "BBB": FakeTicker({"freeCashflow": 5.0, "marketCap": 0})})
    df = fetch_fundamentals(["AAA", "BBB"], ticker_factory=f)
    assert np.isnan(df.loc["AAA", "fcf_yield"])
    assert np.isnan(df.loc["BBB", "fcf_yield"])


def test_one_failing_ticker_does_not_abort_batch(capsys):
    f = factory({"GOOD": FakeTicker({"trailingPE": 10.0}),
                 "BAD": FakeTicker(boom=True)})
    df = fetch_fundamentals(["BAD", "GOOD"], ticker_factory=f)
    assert set(df.index) == {"BAD", "GOOD"}
    assert df.loc["GOOD", "pe"] == 10.0
    assert df.loc[["BAD"], ["pe", "ev_ebitda", "fcf_yield"]].isna().all().all()
    assert "[fetch]" in capsys.readouterr().out
    an = fetch_analyst(["BAD", "GOOD"], ticker_factory=f)
    assert set(an.index) == {"BAD", "GOOD"}
    assert an.loc["BAD"].isna().all()
    e = fetch_earnings(["BAD"], ticker_factory=f)
    assert pd.isna(e["BAD"])


def test_analyst_rec_trend_computed_from_summary():
    summary = pd.DataFrame({
        "period": ["0m", "-1m", "-2m", "-3m"],
        "strongBuy": [5, 4, 3, 2],
        "buy": [5, 5, 5, 3],
        "hold": [10, 10, 10, 10],
        "sell": [0, 0, 0, 5],
        "strongSell": [0, 0, 0, 0],
    })
    info = {"recommendationMean": 2.1, "targetMeanPrice": 120.0, "currentPrice": 100.0}
    f = factory({"AAA": FakeTicker(info, summary=summary)})
    df = fetch_analyst(["AAA"], ticker_factory=f)
    assert list(df.columns) == ["rec_mean", "target_gap", "rec_trend"]
    assert df.loc["AAA", "rec_mean"] == 2.1
    assert abs(df.loc["AAA", "target_gap"] - 0.2) < 1e-9
    assert abs(df.loc["AAA", "rec_trend"] - (10 / 20 - 5 / 20)) < 1e-9


def test_analyst_target_gap_falls_back_to_regular_market_price():
    info = {"targetMeanPrice": 110.0, "regularMarketPrice": 100.0}
    df = fetch_analyst(["A"], ticker_factory=factory({"A": FakeTicker(info)}))
    assert abs(df.loc["A", "target_gap"] - 0.1) < 1e-9
    assert np.isnan(df.loc["A", "rec_trend"])


def test_earnings_unknown_is_nat():
    future = pd.Timestamp.now() + pd.Timedelta(days=10)
    f = factory({"NONE": FakeTicker(calendar={}),
                 "OK": FakeTicker(calendar={"Earnings Date": [future.date()]})})
    s = fetch_earnings(["NONE", "OK"], ticker_factory=f)
    assert pd.isna(s["NONE"])
    assert s["OK"] == pd.Timestamp(future.date())
    assert s["OK"].tzinfo is None
