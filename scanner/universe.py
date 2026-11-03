from __future__ import annotations

import io

import pandas as pd
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WINDOW = 20


def normalise_ticker(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def fetch_sp500() -> pd.DataFrame:
    resp = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0 (scanner)"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    return pd.DataFrame(
        {
            "ticker": table["Symbol"].astype(str).map(normalise_ticker),
            "sector": table["GICS Sector"],
        }
    ).reset_index(drop=True)


def liquidity_filter(
    prices: dict[str, pd.DataFrame],
    min_dollar_vol: float,
    min_price: float,
    window: int = WINDOW,
) -> tuple[list[str], dict[str, str]]:
    eligible: list[str] = []
    rejected: dict[str, str] = {}
    for ticker, df in prices.items():
        if len(df) < window:
            rejected[ticker] = "short_history"
            continue
        if float(df["Close"].iloc[-1]) < min_price:
            rejected[ticker] = "price"
            continue
        dollar_vol = float((df["Close"] * df["Volume"]).tail(window).mean())
        if dollar_vol < min_dollar_vol:
            rejected[ticker] = "dollar_volume"
            continue
        eligible.append(ticker)
    return eligible, rejected
