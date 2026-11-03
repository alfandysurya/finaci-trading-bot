# Telegram Stock Scanner — Design Spec

Date: 2026-11-03
Status: Draft for user review (no code written)

## 1. Purpose

A Telegram bot that surfaces the top 3 "underrated" US stocks each week, each with
technical, fundamental, sentiment and risk scores, a suggested quantity and
stop-loss, plus performance/sell alerts and a win/loss record. It is **PAPER ONLY**
until a live forward paper record justifies more (see section 9).

The user's 5-10% net/month figure is a dashboard target. It must not influence
position size. Tested results in this repo are 0.9-1.8%/month
(see memory: feasibility-review-2026-10).

## 2. Agreed decisions (from design dialogue)

| Topic | Decision |
|---|---|
| "Underrated" | Blend: value (cheap vs fundamentals) + improving trend + price trigger, ranked via the four scores |
| Universe | S&P 500 with hard liquidity filter: 20-day avg dollar volume >= $50M, price >= $5 |
| Cadence | Weekly top 3; daily checks on open positions only (stop/sell alerts) |
| Record | Every pick auto-paper-traded (entry = next open) plus a "took it" flag set by the user; results split by config version |
| Indicators | Profiles (Balanced / Value-heavy / Momentum-heavy) as presets of ~6-8 toggles; any tweak = "Custom (based on X)"; weights fixed per profile |
| Sentiment | Analyst consensus / revisions from yfinance only (no news in v1) |
| Hosting | GitHub Actions on a schedule; button taps are processed at the next run |
| Mirror | One-way push to Google Sheets via service account; repo is source of truth |
| Risk | 1% risk per trade; hold 2-8 weeks; 20% position cap; ~5% total open risk; pause new buys at -20% from peak; $30 minimum order |
| Data | Free only (yfinance for prices, fundamentals, analyst data) |

## 3. Architecture

New package `scanner/` beside `trader/`. It shares only data-fetch helpers with
`trader/`; it has its own config, cache and ledger so the unvalidated scanner
cannot affect the SMA trader or its validation gate.

Four units, each with one purpose:

1. **Universe** — loads the S&P 500 constituent list, fetches prices, applies the
   liquidity filter. Output: list of eligible tickers + price/volume frames.
2. **Scorer** — computes technical, fundamental, sentiment and risk scores (each a
   0-100 percentile rank across the eligible universe), applies gates, blends via
   the active profile, selects the top 3, and computes stop and quantity.
3. **Ledger** — paper positions and win/loss record, stored as plain files in git.
4. **Notifier** — Telegram messages and button handling, plus the Google Sheets mirror.

Workflows:
- Weekly job: Universe -> Scorer -> Ledger (open paper positions) -> Notifier.
- Daily job: Ledger (mark-to-market, stop/sell checks) -> Notifier. No scoring.

Indicators are plain pandas/numpy in `scanner/indicators.py` (no TA-Lib), matching
the existing `trader/indicators.py` style, so every score is reproducible offline.

## 4. Scoring and selection

Gates (a stock failing any gate is not a candidate):
- Liquidity filter (section 2).
- Trend gate: close above SMA(200) and positive 3-month momentum. Prevents value traps.
- Earnings blackout: no new pick within 5 trading days before the next earnings date.
- Data completeness: any required input missing or stale -> skip and report, never guess.

Scores (each 0-100, percentile rank across eligible universe):
- Technical: SMA trend, RSI, MACD, volume trend (toggleable).
- Fundamental: valuation (P/E, EV/EBITDA, FCF yield) vs peers/own history (toggleable).
- Sentiment: analyst recommendation trend and target-price gap.
- Risk: realised volatility, drawdown from high, ATR as a % of price (lower risk = higher score).

Blend: profile weights over the four scores. Top 3 chosen with at most 2 per sector
(target 1-2). Each pick message lists its top reasons.

Config version: hash of (profile, toggle set, weights). A change creates a new
version; ledger results are reported per version and never pooled across versions.

## 5. Sizing and stops

- Stop = entry - 2 x ATR(14).
- Quantity = (1% of account) / (entry - stop), then capped at 20% of account.
- If the resulting order is under $30, the pick is reported as "below minimum" and
  not opened.
- Total open risk target about 5%; if adding a pick would exceed it, it is skipped
  and reported.
- New buys pause when account equity is 20% below its peak; alerts continue.
- Fractional shares cannot use good-till-cancelled orders on Pluang, so the stop
  is a monitored alert level, not a resting order. Behaviour to be tested.

## 6. Ledger

Plain files in git (CSV/parquet). One row per pick: ticker, pick date, entry
(next open), stop, quantity, config version, full score snapshot, data timestamp,
`took_it` flag, exit date/price/reason, P&L, win/loss. Runs are idempotent: re-running
a week cannot duplicate picks (key = pick date + ticker + config version).

The record is forward-only. It is never backfilled, because free data uses today's
S&P 500 membership and would flatter any history.

## 7. Notifier

Telegram messages:
- Weekly: top 3 with scores, reasons, entry/stop/quantity, profile and config version.
- Daily: stop hit, sell signal, open-position P&L, win/loss summary.
- Failures: any job error or data problem is sent as an alert.

Controls (inline buttons / commands): choose profile, toggle indicators, mark
"took it". Taps are processed at the next run (up to ~1 day delay). Telegram cannot
place Pluang orders; the user orders via Claude or the Pluang app during US market
hours (Pluang's agent cannot trade when the exchange is closed).

Google Sheets mirror runs last and is one-way: tabs for picks, open positions, win/loss
by config version, latest scores. A mirror failure never blocks Telegram or the ledger.
A separate user-notes tab is never written by the job.

## 7b. Historical data archive

Added 2026-11-03 at the user's request: record all the data the scanner sees, every
run, so a true point-in-time dataset accumulates from day one.

- What is archived per run: the S&P 500 constituent list as fetched, daily OHLCV for
  the eligible universe, the fundamentals fields used, analyst consensus/target/revision
  fields, earnings dates, and the computed scores for every ticker (not just the top 3).
  Each snapshot carries a fetch timestamp and source name.
- Append-only: snapshots are dated, never overwritten or edited. Corrections are new
  snapshots.
- Format: compressed parquet, one folder per run date (`YYYY/MM/DD/`).
- Location: a Google Drive folder, written by the same Google service account used for
  the Sheets mirror (the user shares the folder with the service account). Git holds
  only the ledger and a small manifest (run date, file names, row counts, checksums), so
  the repo stays small. GitHub Actions artifacts are not used as storage (they expire).
- Failure rule: an archive failure is alerted but does not block Telegram or the ledger.
  A run with a failed archive is marked in the manifest so gaps are visible.
- Purpose: accumulating this data is what makes a later, honest point-in-time backtest
  possible. It does not change the "no backtest in v1" decision.
- Setup risk to verify first: service accounts have little or no Drive storage of their
  own, so uploads may fail unless the folder lives in the user's Drive with enough free
  space or in a shared drive. Confirm this with a test upload before relying on it.
  Estimated volume is a few MB per run compressed (to be measured on the first runs).

## 8. Testing

- Unit tests for each indicator and score on fixed fixtures.
- Ledger tests: sizing, stop, caps, win/loss maths, idempotency.
- Gate tests: trend, earnings blackout, liquidity, sector cap.
- Offline end-to-end dry run on cached data with Telegram and Sheets stubbed.

### No backtest in v1

Decided 2026-11-03: no backtest. Free data cannot give point-in-time fundamentals or
analyst history, and S&P 500 membership history (rebuildable from Wikipedia's change
table) still lacks prices for many delisted names, so any backtest would be biased
and could tempt parameter tuning. The forward paper record (section 6) is the only
performance evidence. A price-only backtest may be revisited later as a separate spec.

## 9. Go-live criteria (proposed)

The scanner stays paper only. Consider live use only after a forward paper record of
at least 6 months and 30 closed trades in one config version, with net return after
a cost assumption beating SPY on a risk-adjusted basis and drawdown within the -20%
pause rule. Thresholds to be confirmed by the user before that review.

## 10. Out of scope (v1)

Backtesting (including historical S&P membership reconstruction); two-way Sheets sync; live order placement; news sentiment; instant button handling
(needs an always-on host); small caps; backfilled history.

## 11. Open items for review

- Liquidity thresholds ($50M / $5) are proposed values.
- Earnings blackout length (5 trading days) and ATR multiple (2x) are proposed values.
- Cost assumption for the go-live comparison is not yet set.
