import numpy as np
import pandas as pd
import pytest

from scanner import fetch as fetch_mod
from scanner.fetch import fetch_analyst, fetch_earnings, fetch_fundamentals


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Tests that do not pass their own sleep must never really wait."""
    monkeypatch.setattr(fetch_mod, "_sleep", lambda seconds: None)


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


# ----------------------------------------------------------------- throttle / retry
class YFRateLimitError(Exception):
    """Same class name as yfinance's rate-limit error."""


class Flaky:
    """Ticker whose .info raises the queued errors first, then returns `info`."""

    def __init__(self, errors, info=None):
        self.errors = list(errors)
        self._info = info or {"trailingPE": 10.0}

    def _check(self):
        if self.errors:
            raise self.errors.pop(0)

    @property
    def info(self):
        self._check()
        return self._info

    @property
    def recommendations_summary(self):
        return None

    @property
    def calendar(self):
        self._check()
        return {}


class Counting:
    def __init__(self, mapping):
        self.mapping, self.calls = mapping, []

    def __call__(self, t):
        self.calls.append(t)
        return self.mapping[t]


class SleepLog:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def test_min_interval_sleeps_between_tickers_only():
    sleep = SleepLog()
    f = factory({t: FakeTicker({"trailingPE": 1.0}) for t in "ABC"})
    fetch_fundamentals(list("ABC"), ticker_factory=f, sleep=sleep, min_interval=0.25)
    assert sleep.calls == [0.25, 0.25]


def test_default_min_interval_is_quarter_second():
    sleep = SleepLog()
    f = factory({t: FakeTicker({"trailingPE": 1.0}) for t in "AB"})
    fetch_earnings(list("AB"), ticker_factory=f, sleep=sleep)
    assert sleep.calls == [0.25]


def test_rate_limit_is_retried_then_succeeds():
    sleep = SleepLog()
    tk = Flaky([YFRateLimitError("Too Many Requests. Rate limited."),
                RuntimeError("429 Too Many Requests")])
    df = fetch_fundamentals(["A"], ticker_factory=lambda t: tk, sleep=sleep, min_interval=0)
    assert sleep.calls == [5, 15]
    assert df.loc["A", "pe"] == 10.0


def test_other_errors_are_not_retried(capsys):
    sleep = SleepLog()
    make = Counting({"A": Flaky([ValueError("bad json")])})
    df = fetch_fundamentals(["A"], ticker_factory=make, sleep=sleep, min_interval=0)
    assert sleep.calls == []
    assert make.calls == ["A"]
    assert np.isnan(df.loc["A", "pe"])
    assert "fundamentals failed for A: ValueError: bad json" in capsys.readouterr().out


def test_rate_limit_after_three_retries_gives_nan():
    sleep = SleepLog()
    make = Counting({"A": Flaky([YFRateLimitError("x")] * 4)})
    df = fetch_fundamentals(["A"], ticker_factory=make, sleep=sleep, min_interval=0)
    assert sleep.calls == [5, 15, 45]
    assert len(make.calls) == 4
    assert np.isnan(df.loc["A", "pe"])


def test_circuit_breaker_skips_remaining_after_eight_rate_limited(capsys):
    tickers = [f"R{i}" for i in range(8)] + ["X1", "X2", "X3"]
    mapping = {t: Flaky([YFRateLimitError("Too Many Requests")] * 99) for t in tickers}
    make = Counting(mapping)
    sleep = SleepLog()
    s = fetch_earnings(tickers, ticker_factory=make, sleep=sleep, min_interval=0)
    assert not any(t.startswith("X") for t in make.calls)
    assert len(make.calls) == 8 * 4
    assert s.isna().all() and list(s.index) == tickers
    out = capsys.readouterr().out
    assert out.count("[fetch] rate limited: skipping 3 remaining earnings") == 1


def test_circuit_breaker_needs_consecutive_rate_limits():
    rl = lambda: Flaky([YFRateLimitError("x")] * 99)  # noqa: E731
    tickers = [f"R{i}" for i in range(7)] + ["OK"] + [f"S{i}" for i in range(7)] + ["LAST"]
    mapping = {t: rl() for t in tickers}
    mapping["OK"] = FakeTicker({"trailingPE": 3.0})
    mapping["LAST"] = FakeTicker({"trailingPE": 4.0})
    make = Counting(mapping)
    df = fetch_fundamentals(tickers, ticker_factory=make, sleep=SleepLog(), min_interval=0)
    assert "LAST" in make.calls
    assert df.loc["OK", "pe"] == 3.0 and df.loc["LAST", "pe"] == 4.0


def test_breaker_fills_analyst_rows_with_nan():
    tickers = [f"R{i}" for i in range(9)]
    mapping = {t: Flaky([YFRateLimitError("x")] * 99) for t in tickers}
    df = fetch_analyst(tickers, ticker_factory=Counting(mapping), sleep=SleepLog(),
                       min_interval=0)
    assert list(df.index) == tickers
    assert df.isna().all().all()


# ----------------------------------------------------------------- shared rate-limit state
def test_tripped_breaker_makes_later_fetches_skip_long_waits(capsys):
    from scanner.fetch import RateLimitState

    state = RateLimitState()
    assert state.tripped is False
    limited = [f"R{i}" for i in range(10)]
    make = Counting({t: Flaky([YFRateLimitError("x")] * 99) for t in limited})
    first = SleepLog()
    fetch_earnings(limited, ticker_factory=make, sleep=first, min_interval=0.25,
                   rate_state=state)
    assert state.tripped is True
    assert 45 in first.calls

    # Next fetch in the same run: one attempt per ticker, only the min-interval pauses,
    # and the breaker still bails out after 8 consecutive rate-limited tickers.
    make2 = Counting({t: Flaky([YFRateLimitError("x")] * 99) for t in limited})
    second = SleepLog()
    df = fetch_fundamentals(limited, ticker_factory=make2, sleep=second, min_interval=0.25,
                            rate_state=state)
    assert set(second.calls) == {0.25}
    assert make2.calls == limited[:8]
    assert df["pe"].isna().all()
    assert "skipping 2 remaining fundamentals" in capsys.readouterr().out


def test_breaker_trip_on_last_ticker_marks_state():
    from scanner.fetch import RateLimitState

    state = RateLimitState()
    tickers = [f"R{i}" for i in range(8)]
    make = Counting({t: Flaky([YFRateLimitError("x")] * 99) for t in tickers})
    fetch_earnings(tickers, ticker_factory=make, sleep=SleepLog(), min_interval=0,
                   rate_state=state)
    assert state.tripped is True


def test_no_retry_mode_still_fetches_good_tickers():
    from scanner.fetch import RateLimitState

    state = RateLimitState(tripped=True)
    f = factory({"A": FakeTicker({"trailingPE": 7.0})})
    sleep = SleepLog()
    an = fetch_analyst(["A"], ticker_factory=f, sleep=sleep, rate_state=state)
    df = fetch_fundamentals(["A"], ticker_factory=f, sleep=sleep, rate_state=state)
    assert df.loc["A", "pe"] == 7.0 and list(an.index) == ["A"]
    assert sleep.calls == []


def test_fetch_signatures_stay_backward_compatible():
    import inspect

    f = factory({"A": FakeTicker({"trailingPE": 2.0}, calendar={})})
    assert fetch_fundamentals(["A"], f).loc["A", "pe"] == 2.0  # positional factory
    assert list(fetch_analyst(["A"], f).index) == ["A"]
    assert pd.isna(fetch_earnings(["A"], f)["A"])
    for fn in (fetch_fundamentals, fetch_analyst, fetch_earnings):
        params = inspect.signature(fn).parameters
        assert list(params)[:2] == ["tickers", "ticker_factory"]
        assert params["ticker_factory"].default is None
        for name in ("sleep", "min_interval", "retry_waits", "breaker_after", "rate_state"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params["rate_state"].default is None


# ----------------------------------------------------------------- retry-wait budget
def test_default_retry_budget_is_480_seconds():
    from scanner.fetch import RateLimitState

    assert RateLimitState().retry_budget == 480


def test_on_and_off_refusals_exhaust_the_budget_then_stop_retrying(capsys):
    from scanner.fetch import RateLimitState

    state = RateLimitState(retry_budget=100)
    tickers, mapping = [], {}
    for i in range(6):  # refused, ok, refused, ok, ...: never 8 in a row
        tickers += [f"R{i}", f"OK{i}"]
        mapping[f"R{i}"] = Flaky([YFRateLimitError("x")] * 99)
        mapping[f"OK{i}"] = FakeTicker({"trailingPE": 1.0})
    make = Counting(mapping)
    sleep = SleepLog()
    df = fetch_fundamentals(tickers, ticker_factory=make, sleep=sleep, min_interval=0.25,
                            rate_state=state)

    waits = [s for s in sleep.calls if s != 0.25]
    assert sum(waits) <= 100
    assert waits == [5, 15, 45, 5, 15]  # R1's 45 s wait would exceed the budget
    assert state.tripped is True
    assert all(make.calls.count(f"R{i}") == 1 for i in range(2, 6))  # no retries after
    assert "skipping" not in capsys.readouterr().out  # consecutive breaker never tripped
    assert (df.loc[[f"OK{i}" for i in range(6)], "pe"] == 1.0).all()


def test_retry_budget_is_shared_across_fetches():
    from scanner.fetch import RateLimitState

    state = RateLimitState(retry_budget=70)
    first = SleepLog()
    fetch_earnings(["A"], ticker_factory=Counting({"A": Flaky([YFRateLimitError("x")] * 9)}),
                   sleep=first, min_interval=0, rate_state=state)
    assert first.calls == [5, 15, 45] and state.tripped is False
    second = SleepLog()
    make = Counting({"B": Flaky([YFRateLimitError("x")] * 9)})
    fetch_fundamentals(["B"], ticker_factory=make, sleep=second, min_interval=0,
                       rate_state=state)
    assert second.calls == [5]  # 65 + 5 = 70 fits; the next 15 s would not
    assert len(make.calls) == 2
    assert state.tripped is True
