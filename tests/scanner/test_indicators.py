import numpy as np
import pandas as pd

from scanner.indicators import atr, drawdown_from_high, latest_metrics

KEYS = {"close", "sma50", "sma200", "rsi14", "macd_hist", "vol_trend", "ret_63d",
        "atr14", "atr_pct", "drawdown", "vol_20d_ann", "dollar_vol_20d"}


def _frame(n, close=10.0, volume=1e6, half_range=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + half_range, "Low": c - half_range,
                         "Close": c, "Volume": float(volume)}, index=idx)


def test_atr_constant_range():
    df = _frame(60)
    assert atr(df).iloc[-1] == 2.0


def test_atr_uses_prior_close_gap():
    df = _frame(40, half_range=0.0)
    df.loc[df.index[20:], ["High", "Low", "Close"]] = 15.0
    assert atr(df, 1).iloc[20] == 5.0


def test_drawdown_zero_at_new_high_and_negative_after_drop():
    close = pd.Series([1.0, 2.0, 3.0, 2.25])
    dd = drawdown_from_high(close)
    assert dd.iloc[2] == 0.0
    assert dd.iloc[3] == -0.25
    assert (dd <= 0).all()


def test_latest_metrics_nan_when_history_short():
    m = latest_metrics(_frame(30))
    assert set(m) == KEYS
    assert np.isnan(m["sma200"])
    assert m["close"] == 10.0


def test_dollar_volume_is_price_times_volume_mean():
    m = latest_metrics(_frame(300, close=10.0, volume=1e6))
    assert m["dollar_vol_20d"] == 1e7
    assert all(isinstance(v, float) for v in m.values())
