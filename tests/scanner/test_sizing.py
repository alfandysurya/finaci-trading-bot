import math

import pytest

from scanner.sizing import buys_paused, open_risk_ok, size_position, stop_price


def test_size_risks_one_percent():
    r = size_position(1000, 100, 90)
    assert r["status"] == "ok"
    assert r["quantity"] == pytest.approx(1.0)
    assert r["risk_amount"] == pytest.approx(10.0)
    assert r["value"] == pytest.approx(100.0)


def test_size_capped_at_twenty_percent():
    r = size_position(1000, 100, 99)  # uncapped value would be 1000
    assert r["value"] == pytest.approx(200.0)
    assert r["quantity"] == pytest.approx(2.0)
    assert r["risk_amount"] == pytest.approx(2.0)
    assert r["status"] == "ok"


def test_size_below_minimum_reported():
    r = size_position(1000, 10, 9.5, risk_pct=0.001)  # qty 2 -> value 20
    assert r["status"] == "below_minimum"
    assert r["value"] == pytest.approx(20.0)
    assert r["quantity"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "args",
    [
        (0, 100, 90),
        (-5, 100, 90),
        (1000, 100, 100),
        (1000, 100, 110),
        (1000, 0, -1),
        (1000, 100, 0),
        (1000, 100, -3),
        (float("nan"), 100, 90),
        (1000, float("nan"), 90),
        (1000, 100, float("nan")),
        (float("inf"), 100, 90),
        (1000, float("inf"), 90),
    ],
)
def test_size_invalid_when_equity_nonpositive_or_stop_above_entry(args):
    r = size_position(*args)
    assert r == {"quantity": 0.0, "value": 0.0, "risk_amount": 0.0, "status": "invalid"}


def test_open_risk_limit():
    assert open_risk_ok(1000, [49.0], 2.0) is False
    assert open_risk_ok(1000, [20.0, 20.0], 5.0) is True


def test_open_risk_boundary_exactly_five_percent_passes():
    assert open_risk_ok(1000, [48.0], 2.0) is True
    assert open_risk_ok(1000, [0.1 + 0.2, 49.5], 0.2) is True  # float noise


def test_open_risk_nonpositive_equity():
    assert open_risk_ok(0, [], 0.0) is False
    assert open_risk_ok(-10, [], 1.0) is False


def test_buys_paused_at_minus_twenty():
    assert buys_paused(800, 1000) is True
    assert buys_paused(801, 1000) is False
    assert buys_paused(500, 0) is False


def test_stop_price():
    assert stop_price(100, 5) == pytest.approx(90.0)
    assert stop_price(100, 5, mult=1.5) == pytest.approx(92.5)
    assert math.isnan(stop_price(100, float("nan")))
    assert math.isnan(stop_price(float("nan"), 5))
