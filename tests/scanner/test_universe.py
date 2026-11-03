import pandas as pd

from scanner import universe
from scanner.universe import fetch_sp500, liquidity_filter, normalise_ticker


def make_df(price, volume, bars=20):
    idx = pd.date_range("2026-01-01", periods=bars, freq="B")
    return pd.DataFrame(
        {"Open": price, "High": price, "Low": price, "Close": price, "Volume": volume},
        index=idx,
    )


def test_normalise_ticker_dot_to_dash():
    assert normalise_ticker("BRK.B") == "BRK-B"
    assert normalise_ticker(" bf.b ") == "BF-B"


def test_liquidity_filter_rejects_low_dollar_volume():
    ok, rej = liquidity_filter({"AAA": make_df(100, 1000)}, 50_000_000, 5)
    assert ok == [] and rej == {"AAA": "dollar_volume"}


def test_liquidity_filter_rejects_penny_stock():
    ok, rej = liquidity_filter({"PNY": make_df(4.99, 1e9)}, 50_000_000, 5)
    assert ok == [] and rej == {"PNY": "price"}


def test_liquidity_filter_rejects_short_history():
    ok, rej = liquidity_filter({"NEW": make_df(100, 1e9, bars=19)}, 50_000_000, 5)
    assert ok == [] and rej == {"NEW": "short_history"}


def test_liquidity_filter_accepts_boundary():
    ok, rej = liquidity_filter({"EDG": make_df(5, 10_000_000)}, 50_000_000, 5)
    assert ok == ["EDG"] and rej == {}


def test_fetch_sp500_uses_read_html(monkeypatch):
    table = pd.DataFrame(
        {"Symbol": ["AAPL", "BRK.B"], "GICS Sector": ["Information Technology", "Financials"]}
    )

    class Resp:
        text = "<html></html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(universe.pd, "read_html", lambda *a, **k: [table])
    df = fetch_sp500()
    assert list(df.columns) == ["ticker", "sector"]
    assert df["ticker"].tolist() == ["AAPL", "BRK-B"]
    assert df["sector"].tolist() == ["Information Technology", "Financials"]
