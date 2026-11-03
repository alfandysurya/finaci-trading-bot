import json

import pandas as pd
import pytest

from scanner.ledger import (
    COLUMNS,
    add_picks,
    equity,
    fill_pending,
    mark_took_it,
    process_exits,
    read_ledger,
    summary_by_version,
    write_ledger,
)


def picks(rows: dict) -> pd.DataFrame:
    """rows: {ticker: (close, atr14)} -> picks frame shaped like select_top output."""
    df = pd.DataFrame(
        {
            "close": [v[0] for v in rows.values()],
            "atr14": [v[1] for v in rows.values()],
            "blend": [80.0] * len(rows),
        },
        index=list(rows.keys()),
    )
    df["reasons"] = pd.Series([["strong trend"]] * len(rows), index=df.index, dtype=object)
    return df


def bars(data: list[tuple]) -> pd.DataFrame:
    """data: [(date, open, high, low, close), ...]"""
    idx = pd.to_datetime([d[0] for d in data])
    return pd.DataFrame(
        {
            "Open": [d[1] for d in data],
            "High": [d[2] for d in data],
            "Low": [d[3] for d in data],
            "Close": [d[4] for d in data],
            "Volume": [1_000_000.0] * len(data),
        },
        index=idx,
    )


def empty() -> pd.DataFrame:
    return read_ledger("does/not/exist.csv")


def open_position(prices: dict, ticker="AAA", close=100.0, atr=2.5,
                  pick_date="2026-01-02", fill_on="2026-01-05", eq=10_000.0):
    led = add_picks(empty(), picks({ticker: (close, atr)}), pick_date, "v1", eq)
    led = fill_pending(led, prices, fill_on)
    assert (led["status"] == "open").all()
    return led


# Stop fixture: pick 2026-01-02 close 100, atr 2.5 -> stop 95, qty 20.
def stop_prices(day2_open, day2_low):
    return {
        "AAA": bars([
            ("2026-01-02", 99.0, 100.5, 98.5, 100.0),
            ("2026-01-05", 100.0, 101.0, 98.0, 100.0),
            ("2026-01-06", day2_open, max(day2_open, 98.0), day2_low, 96.0),
        ])
    }


def test_add_picks_idempotent(tmp_path):
    p = picks({"AAA": (100.0, 2.5), "BBB": (50.0, 1.0)})
    led = add_picks(empty(), p, "2026-01-02", "v1", 10_000.0)
    assert len(led) == 2
    row = led[led["ticker"] == "AAA"].iloc[0]
    assert row["status"] == "pending"
    assert row["stop"] == pytest.approx(95.0)
    assert row["quantity"] == pytest.approx(20.0)
    assert not row["took_it"]
    snap = json.loads(row["score_snapshot"])
    assert snap["atr14"] == pytest.approx(2.5)
    assert snap["close"] == pytest.approx(100.0)

    path = tmp_path / "state" / "ledger.csv"
    write_ledger(led, path)
    reread = read_ledger(path)
    again = add_picks(reread, p, pd.Timestamp("2026-01-02"), "v1", 5_000.0)
    assert len(again) == 2
    pd.testing.assert_frame_equal(again.reset_index(drop=True), reread.reset_index(drop=True))

    # a different config version is a different key
    other = add_picks(again, p, "2026-01-02", "v2", 10_000.0)
    assert len(other) == 4


def test_below_minimum_pick_is_skipped_not_opened():
    led = add_picks(empty(), picks({"AAA": (100.0, 2.5)}), "2026-01-02", "v1", 100.0)
    row = led.iloc[0]
    assert row["status"] == "skipped"
    assert row["exit_reason"] == "below_minimum"
    prices = stop_prices(97.0, 94.0)
    led = fill_pending(led, prices, "2026-01-06")
    assert led.iloc[0]["status"] == "skipped"
    led, events = process_exits(led, prices, "2026-01-06")
    assert led.iloc[0]["status"] == "skipped"
    assert events == []


def test_invalid_sizing_is_skipped():
    led = add_picks(empty(), picks({"AAA": (100.0, float("nan"))}), "2026-01-02", "v1", 10_000.0)
    assert led.iloc[0]["status"] == "skipped"
    assert led.iloc[0]["exit_reason"] == "invalid"


def test_fill_pending_uses_next_open_not_pick_day_close():
    prices = stop_prices(97.0, 96.5)
    led = add_picks(empty(), picks({"AAA": (100.0, 2.5)}), "2026-01-02", "v1", 10_000.0)

    same_day = fill_pending(led, prices, "2026-01-02")
    assert same_day.iloc[0]["status"] == "pending"
    assert same_day.iloc[0]["entry_date"] == ""

    filled = fill_pending(led, prices, "2026-01-06")
    row = filled.iloc[0]
    assert row["status"] == "open"
    assert row["entry_date"] == "2026-01-05"
    assert row["entry_price"] == pytest.approx(100.0)  # 2026-01-05 Open, not 01-02 Close
    assert row["stop"] == pytest.approx(95.0)
    assert row["quantity"] == pytest.approx(20.0)

    # no price data for the ticker -> stays pending
    assert fill_pending(led, {}, "2026-01-06").iloc[0]["status"] == "pending"


def test_fill_pending_uses_first_open_after_pick_with_distinct_open():
    prices = {"AAA": bars([
        ("2026-01-02", 99.0, 100.5, 98.5, 100.0),
        ("2026-01-05", 101.5, 102.0, 101.0, 101.8),
        ("2026-01-06", 103.0, 104.0, 102.0, 103.5),
    ])}
    led = add_picks(empty(), picks({"AAA": (100.0, 2.5)}), "2026-01-02", "v1", 10_000.0)
    row = fill_pending(led, prices, "2026-01-06").iloc[0]
    assert row["entry_date"] == "2026-01-05"
    assert row["entry_price"] == pytest.approx(101.5)


def test_stop_hit_exits_at_stop_when_bar_trades_through():
    prices = stop_prices(day2_open=97.0, day2_low=94.0)
    led = open_position(prices)
    led, events = process_exits(led, prices, "2026-01-06")
    row = led.iloc[0]
    assert row["status"] == "closed"
    assert row["exit_reason"] == "stop"
    assert row["exit_date"] == "2026-01-06"
    assert row["exit_price"] == pytest.approx(95.0)
    assert row["pnl"] == pytest.approx(-100.0)
    assert row["outcome"] == "loss"
    assert events == [{"ticker": "AAA", "pick_date": "2026-01-02", "kind": "stop",
                       "price": pytest.approx(95.0), "date": "2026-01-06"}]


def test_gap_down_through_stop_exits_at_open():
    prices = stop_prices(day2_open=93.0, day2_low=92.0)
    led = open_position(prices)
    led, events = process_exits(led, prices, "2026-01-06")
    row = led.iloc[0]
    assert row["exit_price"] == pytest.approx(93.0)
    assert row["pnl"] == pytest.approx(-140.0)
    assert row["exit_reason"] == "stop"
    assert events[0]["price"] == pytest.approx(93.0)


def test_stop_checked_on_entry_bar():
    prices = {"AAA": bars([
        ("2026-01-02", 99.0, 100.5, 98.5, 100.0),
        ("2026-01-05", 99.0, 99.5, 94.5, 96.0),
    ])}
    led = open_position(prices)
    led, events = process_exits(led, prices, "2026-01-05")
    row = led.iloc[0]
    assert row["status"] == "closed"
    assert row["exit_price"] == pytest.approx(95.0)
    assert row["pnl"] == pytest.approx((95.0 - 99.0) * 20.0)


# Sell-signal fixture: 60 flat bars at 100, pick on bar 59, entry bar 60 at Open 100,
# atr 10 -> stop 80, qty 5. Bar 61 closes 97 < SMA50 (99.94) -> alert; bar 62 Open 96.
def sell_prices():
    dates = pd.bdate_range("2026-01-01", periods=63)
    data = [(d, 100.0, 100.0, 100.0, 100.0) for d in dates[:61]]
    data.append((dates[61], 100.0, 100.0, 97.0, 97.0))
    data.append((dates[62], 96.0, 96.5, 95.0, 96.0))
    return {"AAA": bars(data)}, [d.strftime("%Y-%m-%d") for d in dates]


def test_sell_signal_closes_at_next_open():
    prices, d = sell_prices()
    led = open_position(prices, atr=10.0, pick_date=d[59], fill_on=d[60])
    assert led.iloc[0]["quantity"] == pytest.approx(5.0)

    led, events = process_exits(led, prices, d[61])
    row = led.iloc[0]
    assert row["status"] == "open"
    assert row["pending_exit"] == "sell_signal"
    assert events == [{"ticker": "AAA", "pick_date": d[59], "kind": "sell_signal",
                       "price": pytest.approx(97.0), "date": d[61]}]

    led, events = process_exits(led, prices, d[62])
    row = led.iloc[0]
    assert row["status"] == "closed"
    assert row["exit_reason"] == "sell_signal"
    assert row["exit_date"] == d[62]
    assert row["exit_price"] == pytest.approx(96.0)
    assert row["pnl"] == pytest.approx(-20.0)
    assert row["pending_exit"] == ""
    assert events == [{"ticker": "AAA", "pick_date": d[59], "kind": "closed",
                       "exit_reason": "sell_signal", "price": pytest.approx(96.0),
                       "date": d[62]}]


def test_sell_signal_and_close_in_one_call_emit_both_events():
    prices, d = sell_prices()
    led = open_position(prices, atr=10.0, pick_date=d[59], fill_on=d[60])
    led, events = process_exits(led, prices, d[62])
    assert [e["kind"] for e in events] == ["sell_signal", "closed"]
    assert led.iloc[0]["exit_price"] == pytest.approx(96.0)


def time_prices():
    # Rising: close = 100 + 0.5*i, open = close - 0.25, low = open - 0.25.
    dates = pd.bdate_range("2026-01-02", "2026-03-04")
    data = []
    for i, dt in enumerate(dates):
        c = 100.0 + 0.5 * i
        data.append((dt, c - 0.25, c + 0.25, c - 0.5, c))
    return {"AAA": bars(data)}


def test_time_exit_after_56_days():
    prices = time_prices()
    led = open_position(prices)  # entry 2026-01-05 at Open 100.25, stop 95
    assert led.iloc[0]["entry_price"] == pytest.approx(100.25)

    led, events = process_exits(led, prices, "2026-02-27")  # 53 days held
    assert led.iloc[0]["pending_exit"] == ""
    assert events == []

    led, events = process_exits(led, prices, "2026-03-02")  # 56 days held
    assert led.iloc[0]["status"] == "open"
    assert led.iloc[0]["pending_exit"] == "time_exit"
    assert events[0]["kind"] == "time_exit"
    assert events[0]["date"] == "2026-03-02"
    assert events[0]["price"] == pytest.approx(120.5)  # 2026-03-02 is bar 41 -> close 120.5

    led, events = process_exits(led, prices, "2026-03-04")
    row = led.iloc[0]
    assert row["status"] == "closed"
    assert row["exit_reason"] == "time_exit"
    assert row["exit_date"] == "2026-03-03"
    # 2026-03-03 is bar index 42 -> close 121.0, open 120.75
    assert row["exit_price"] == pytest.approx(120.75)
    assert row["pnl"] == pytest.approx((120.75 - 100.25) * 20.0)
    assert [e["kind"] for e in events] == ["closed"]


def test_outcome_win_loss_and_pnl_sign():
    win = time_prices()["AAA"]
    loss = stop_prices(97.0, 94.0)["AAA"]
    prices = {"WIN": win, "LOSS": loss}
    led = add_picks(empty(), picks({"WIN": (100.0, 2.5), "LOSS": (100.0, 2.5)}),
                    "2026-01-02", "v1", 10_000.0)
    led = fill_pending(led, prices, "2026-01-05")
    led, _ = process_exits(led, prices, "2026-03-04")
    w = led[led["ticker"] == "WIN"].iloc[0]
    lo = led[led["ticker"] == "LOSS"].iloc[0]
    assert w["outcome"] == "win" and w["pnl"] > 0
    assert w["pnl"] == pytest.approx(410.0)
    assert lo["outcome"] == "loss" and lo["pnl"] < 0
    assert lo["pnl"] == pytest.approx(-100.0)


def closed_row(version, ticker, pnl, pick_date="2026-01-02"):
    return {
        "pick_date": pick_date, "ticker": ticker, "config_version": version,
        "status": "closed", "entry_date": "2026-01-05", "entry_price": 100.0,
        "stop": 95.0, "quantity": 1.0, "took_it": False, "score_snapshot": "{}",
        "data_timestamp": pick_date, "exit_date": "2026-01-10",
        "exit_price": 100.0 + pnl, "exit_reason": "stop", "pnl": pnl,
        "outcome": "win" if pnl > 0 else "loss", "pending_exit": "",
    }


def test_summary_never_pools_versions():
    rows = [closed_row("v1", "AAA", 10.0), closed_row("v1", "BBB", -5.0),
            closed_row("v2", "CCC", 20.0)]
    led = pd.concat([empty(), pd.DataFrame(rows)], ignore_index=True)
    # a non-closed row must not count
    led = add_picks(led, picks({"DDD": (100.0, 2.5)}), "2026-01-09", "v2", 10_000.0)
    s = summary_by_version(led).set_index("config_version")
    assert sorted(s.index) == ["v1", "v2"]
    assert s.loc["v1", "trades"] == 2
    assert s.loc["v1", "wins"] == 1
    assert s.loc["v1", "losses"] == 1
    assert s.loc["v1", "win_rate"] == pytest.approx(0.5)
    assert s.loc["v1", "total_pnl"] == pytest.approx(5.0)
    assert s.loc["v2", "trades"] == 1
    assert s.loc["v2", "total_pnl"] == pytest.approx(20.0)


def test_mark_took_it_sets_flag_only():
    p = picks({"AAA": (100.0, 2.5), "BBB": (50.0, 1.0)})
    led = add_picks(empty(), p, "2026-01-02", "v1", 10_000.0)
    led = add_picks(led, p, "2026-01-02", "v2", 10_000.0)
    led = add_picks(led, p, "2026-01-09", "v1", 10_000.0)
    out = mark_took_it(led, "AAA", "2026-01-02")
    flagged = out[out["took_it"]]
    assert len(flagged) == 2
    assert set(flagged["config_version"]) == {"v1", "v2"}
    assert (flagged["ticker"] == "AAA").all()
    assert (flagged["pick_date"] == "2026-01-02").all()
    pd.testing.assert_frame_equal(out.drop(columns="took_it"), led.drop(columns="took_it"))
    assert not led["took_it"].any()  # input not mutated


def test_process_exits_idempotent_on_rerun(tmp_path):
    prices, d = sell_prices()
    led = open_position(prices, atr=10.0, pick_date=d[59], fill_on=d[60])
    led, events = process_exits(led, prices, d[61])
    assert len(events) == 1
    path = tmp_path / "ledger.csv"
    write_ledger(led, path)
    led = read_ledger(path)
    again, events = process_exits(led, prices, d[61])
    assert events == []
    pd.testing.assert_frame_equal(again, led)

    closed, events = process_exits(led, prices, d[62])
    assert len(events) == 1
    write_ledger(closed, path)
    closed = read_ledger(path)
    again, events = process_exits(closed, prices, d[62])
    assert events == []
    pd.testing.assert_frame_equal(again, closed)


def test_read_ledger_missing_file_returns_empty_frame(tmp_path):
    led = read_ledger(tmp_path / "missing.csv")
    assert list(led.columns) == COLUMNS
    assert len(led) == 0
    path = tmp_path / "a" / "b" / "ledger.csv"
    write_ledger(led, path)
    back = read_ledger(path)
    assert list(back.columns) == COLUMNS
    assert len(back) == 0
    assert "pending_exit" in COLUMNS


def test_read_preserves_string_keys(tmp_path):
    led = add_picks(empty(), picks({"NA": (100.0, 2.5)}), "2026-01-02", "1e53000", 10_000.0)
    path = tmp_path / "ledger.csv"
    write_ledger(led, path)
    back = read_ledger(path)
    assert back.iloc[0]["ticker"] == "NA"
    assert back.iloc[0]["config_version"] == "1e53000"
    assert back.iloc[0]["pending_exit"] == ""
    assert len(add_picks(back, picks({"NA": (100.0, 2.5)}), "2026-01-02", "1e53000", 1.0)) == 1


def test_equity_includes_unrealised_pnl():
    rows = [closed_row("v1", "AAA", -100.0)]
    led = pd.concat([empty(), pd.DataFrame(rows)], ignore_index=True)
    prices = {"BBB": bars([
        ("2026-01-02", 99.0, 100.5, 98.5, 100.0),
        ("2026-01-05", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-06", 105.0, 111.0, 104.0, 110.0),
    ])}
    # BBB: close 100, atr 10 -> stop 80, qty 5, entry 100 on 2026-01-05
    led = add_picks(led, picks({"BBB": (100.0, 10.0)}), "2026-01-02", "v1", 10_000.0)
    led = fill_pending(led, prices, "2026-01-05")
    assert equity(led, 10_000.0, prices) == pytest.approx(10_000.0 - 100.0 + 5 * 10.0)


def test_stop_wins_over_same_bar_sell_signal():
    # 62 bars; last bar has Low 79 <= stop 80 and Close 97 < SMA50 (99.94).
    dates = pd.bdate_range("2026-01-01", periods=62)
    data = [(dt, 100.0, 100.0, 100.0, 100.0) for dt in dates[:61]]
    data.append((dates[61], 99.0, 100.0, 79.0, 97.0))
    prices = {"AAA": bars(data)}
    d = [dt.strftime("%Y-%m-%d") for dt in dates]
    led = open_position(prices, atr=10.0, pick_date=d[59], fill_on=d[60])
    led, events = process_exits(led, prices, d[61])
    row = led.iloc[0]
    assert row["status"] == "closed"
    assert row["exit_reason"] == "stop"
    assert row["exit_price"] == pytest.approx(80.0)  # min(stop 80, Open 99)
    assert row["pending_exit"] == ""
    assert row["pnl"] == pytest.approx((80.0 - 100.0) * 5.0)
    assert [e["kind"] for e in events] == ["stop"]


def scaled(prices: dict, factor: float) -> dict:
    out = {}
    for t, df in prices.items():
        df = df.copy()
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] * factor
        out[t] = df
    return out


def test_split_adjusted_prices_skip_row_and_report_mismatch():
    nominal = stop_prices(day2_open=97.0, day2_low=96.0)  # no stop in nominal terms
    led = open_position(nominal)  # entry 100 on 2026-01-05, stop 95
    split = scaled(nominal, 0.5)  # 2:1 split: every past bar halved
    out, events = process_exits(led, split, "2026-01-06")
    pd.testing.assert_frame_equal(out, led)
    assert events == [{"ticker": "AAA", "pick_date": "2026-01-02", "kind": "data_mismatch",
                       "price": pytest.approx(100.0), "date": "2026-01-05",
                       "detail": pytest.approx(50.0)}]
    # re-reported every call by design
    _, again = process_exits(out, split, "2026-01-06")
    assert [e["kind"] for e in again] == ["data_mismatch"]


def test_small_dividend_drift_is_processed_normally():
    nominal = stop_prices(day2_open=97.0, day2_low=94.0)
    led = open_position(nominal)
    drift = scaled(nominal, 0.99)  # entry bar Open 99.0 vs stored 100.0 -> 1%
    out, events = process_exits(led, drift, "2026-01-06")
    row = out.iloc[0]
    assert row["status"] == "closed"
    assert row["exit_reason"] == "stop"
    assert row["exit_price"] == pytest.approx(95.0)  # min(stop 95, Open 96.03)
    assert [e["kind"] for e in events] == ["stop"]


def test_sunday_pick_fills_at_monday_open():
    prices = {"AAA": bars([
        ("2026-01-02", 99.0, 100.5, 98.5, 100.0),
        ("2026-01-05", 101.5, 102.0, 101.0, 101.8),
    ])}
    led = add_picks(empty(), picks({"AAA": (100.0, 2.5)}), "2026-01-04", "v1", 10_000.0)
    row = fill_pending(led, prices, "2026-01-05").iloc[0]
    assert row["status"] == "open"
    assert row["entry_date"] == "2026-01-05"
    assert row["entry_price"] == pytest.approx(101.5)


def test_time_exit_flags_at_56_days_not_55_with_daily_bars():
    # Bars on every calendar day; close = 100 + 0.5*i, rising so no SMA50 signal.
    dates = pd.date_range("2026-01-04", "2026-03-03", freq="D")
    data = []
    for i, dt in enumerate(dates):
        c = 100.0 + 0.5 * i
        data.append((dt, c - 0.25, c + 0.25, c - 0.5, c))
    prices = {"AAA": bars(data)}
    led = open_position(prices, pick_date="2026-01-04", fill_on="2026-01-05")
    assert led.iloc[0]["entry_price"] == pytest.approx(100.25)

    led, events = process_exits(led, prices, "2026-03-01")  # 55 days held
    assert led.iloc[0]["pending_exit"] == ""
    assert events == []

    led, events = process_exits(led, prices, "2026-03-02")  # 56 days held
    assert led.iloc[0]["pending_exit"] == "time_exit"
    assert events == [{"ticker": "AAA", "pick_date": "2026-01-04", "kind": "time_exit",
                       "price": pytest.approx(128.5), "date": "2026-03-02"}]
