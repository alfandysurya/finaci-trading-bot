# Telegram Stock Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-only weekly/daily US stock scanner that posts to Telegram, keeps a git-tracked paper ledger, archives every run's raw data to Google Drive, and mirrors results one-way to Google Sheets.

**Architecture:** New package `scanner/` beside `trader/`. Pure functions (indicators, scoring, sizing, ledger rules) take DataFrames and return DataFrames/dicts, so everything is testable offline; network code (yfinance, Wikipedia, Telegram, Drive, Sheets) sits in thin adapters that tests stub. Two GitHub Actions workflows (weekly, daily) call `python -m scanner weekly|daily` and commit the ledger back to git.

**Tech Stack:** Python >= 3.11, pandas, numpy, pyarrow, pyyaml, yfinance, requests (Telegram Bot API), google-api-python-client + google-auth (Drive, Sheets), lxml (Wikipedia table), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-telegram-scanner-design.md`

## Global Constraints

- PAPER ONLY. No code path places or cancels a Pluang order.
- Free data only (yfinance, Wikipedia). No paid APIs.
- Do not touch `trader/` behaviour. `scanner/` may import `trader.indicators.sma/ema/rsi` and `trader.config.ROOT`; nothing in `trader/` imports `scanner/`.
- Liquidity filter: 20-day average dollar volume >= $50,000,000 and price >= $5.
- Trend gate: close > SMA(200) and 3-month (63 trading days) return > 0.
- Earnings blackout: no new pick within 5 trading days before next earnings date.
- Sector cap: at most 2 of the top 3 per sector.
- Stop = entry - 2 x ATR(14). Quantity = 1% of equity / (entry - stop), capped at 20% of equity; below $30 order value -> "below minimum", not opened.
- Total open risk target 5% of equity; new buys paused when equity is 20% below its peak.
- Exits: stop (min(stop, bar Open) on first bar with Low <= stop); close < SMA(50) -> alert, close at next open; 56 calendar days held -> alert, close at next open.
- Scores are 0-100 percentile ranks across the eligible universe. Blend weights per profile are fixed: Balanced 0.30/0.30/0.20/0.20, Value 0.20/0.50/0.15/0.15, Momentum 0.50/0.15/0.15/0.20 (technical/fundamental/sentiment/risk).
- Config version = short hash of (profile, sorted enabled toggles, weights). Ledger results are never pooled across versions.
- Ledger is append-only for picks; idempotency key = (pick_date, ticker, config_version).
- Archive is append-only, dated `YYYY/MM/DD/`, parquet. Archive/Sheets/Drive failures alert but never block Telegram or the ledger.
- Missing or stale data for a ticker -> skip it and report; never fill or guess.
- Secrets (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GOOGLE_SERVICE_ACCOUNT_JSON`, `DRIVE_FOLDER_ID`, `SHEET_ID`) come from environment variables only; never written to files or logs.
- No Claude co-author line in commit messages (user's saved rule).

## Review Focus

Failure modes the spec implies but no obvious task covers; each has a test in the owning task:

1. Tickers with dots (BRK.B, BF.B) must be converted to yfinance form (BRK-B) in the universe, or they silently vanish. (Task 3)
2. Fewer than 3 stocks pass the gates: send the ones that pass, or an explicit "no picks this week" message; never pad with failing stocks. (Task 5, Task 9)
3. A stock gaps down through its stop: exit at the open, not at the stop price. (Task 7)
4. Weekly job re-run the same week: no duplicate ledger rows and no duplicate Telegram message. (Task 7, Task 11)
5. Equity at or below zero, or sizing yields an order under $30: no division error, pick reported as "below minimum". (Task 6)
6. Telegram message over 4096 characters is split, not rejected. (Task 9)

---

## File Structure

- `scanner.yaml` — thresholds, profiles, toggles, paper account (repo root, next to `config.yaml`).
- `scanner/__init__.py`, `scanner/__main__.py` — package + `python -m scanner` entry.
- `scanner/config.py` — load config, `config_version`.
- `scanner/indicators.py` — ATR, MACD histogram, volume trend, drawdown (reuses `trader.indicators`).
- `scanner/universe.py` — S&P 500 list, ticker normalising, liquidity filter.
- `scanner/fetch.py` — yfinance adapters: prices, fundamentals, analyst, earnings, sectors.
- `scanner/scoring.py` — percentile ranks, four scores, gates, blend, top-N selection.
- `scanner/sizing.py` — stop, quantity, open-risk and drawdown-pause rules.
- `scanner/ledger.py` — CSV ledger, state transitions, summaries.
- `scanner/archive.py` — parquet snapshots + manifest.
- `scanner/notify.py` — message formatting, Telegram client, command/settings handling.
- `scanner/google_io.py` — Drive upload and Sheets mirror (service account).
- `scanner/pipeline.py` — `run_weekly`, `run_daily` orchestration; `scanner/cli.py` argument parsing.
- `.github/workflows/scanner-weekly.yml`, `scanner-daily.yml`; `scanner/README.md` setup notes.
- `scanner_state/` (committed) — `ledger.csv`, `settings.json`, `telegram_offset.json`, `manifest.csv`.
- Tests: `tests/scanner/test_*.py`, one file per module.

---

### Task 1: Package scaffold, config and config version

**Files:**
- Create: `scanner.yaml`, `scanner/__init__.py`, `scanner/__main__.py`, `scanner/config.py`, `tests/scanner/test_config.py`
- Modify: `pyproject.toml` (add dependencies `requests`, `lxml`, `google-api-python-client`, `google-auth`), `.gitignore` (add `.env` already present; nothing else)

**Interfaces:**
- Produces: `load_scanner_config(path: str | Path | None = None) -> dict` with keys `universe`, `gates`, `sizing`, `profiles`, `toggles`, `paper_account`, `archive`; `config_version(profile: str, toggles: list[str], weights: dict[str, float]) -> str` (8 hex chars); `active_settings(cfg, settings: dict) -> dict` returning `{"profile", "toggles", "weights", "version", "label"}` where `label` is `"Balanced"` or `"Custom (based on Balanced)"`.

- [ ] **Step 1: Write failing tests** in `tests/scanner/test_config.py`: `test_config_version_stable_and_order_independent` (same toggles in different order -> equal), `test_config_version_changes_with_toggle` (dropping one toggle -> different), `test_label_custom_when_toggles_differ_from_profile_preset` (profile Balanced with one extra toggle off -> `"Custom (based on Balanced)"`), `test_profile_weights_sum_to_one` (each profile in `scanner.yaml`).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_config.py -v` -> FAIL (module missing).
- [ ] **Step 3: Implement** `scanner.yaml` with the Global Constraints values (profiles with preset toggle lists over the eight toggles `sma_trend, rsi, macd, volume, valuation_pe, ev_ebitda, fcf_yield, analyst`; `paper_account.start_equity: 1000  # set to your real account size`), and `scanner/config.py` per the interfaces. Version = first 8 hex of sha1 over a canonical JSON of the inputs.
- [ ] **Step 4: Run** the same command -> PASS.
- [ ] **Step 5: Commit** `git add scanner scanner.yaml tests/scanner pyproject.toml uv.lock && git commit -m "Add scanner package scaffold and config"`.

### Task 2: Scanner indicators

**Files:**
- Create: `scanner/indicators.py`, `tests/scanner/test_indicators.py`

**Interfaces:**
- Consumes: `trader.indicators.sma(s, n)`, `ema(s, n)`, `rsi(close, n)`.
- Produces: `atr(df, n: int = 14) -> pd.Series` (Wilder smoothing, df has High/Low/Close); `macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series`; `volume_trend(volume: pd.Series, short=20, long=60) -> pd.Series` (short mean / long mean); `drawdown_from_high(close: pd.Series, window=252) -> pd.Series` (<= 0); `latest_metrics(df: pd.DataFrame) -> dict` with keys `close, sma50, sma200, rsi14, macd_hist, vol_trend, ret_63d, atr14, atr_pct, drawdown, vol_20d_ann, dollar_vol_20d`, each `float` or `nan` when history is too short.

- [ ] **Step 1: Write failing tests**: `test_atr_constant_range` (flat bars with High-Low = 2 -> atr == 2), `test_atr_uses_prior_close_gap` (a gap up of 5 with tiny bar -> true range 5), `test_drawdown_zero_at_new_high_and_negative_after_drop`, `test_latest_metrics_nan_when_history_short` (30 bars -> `sma200` is nan, no exception), `test_dollar_volume_is_price_times_volume_mean` (fixed close 10, volume 1e6 -> 1e7).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_indicators.py -v` -> FAIL.
- [ ] **Step 3: Implement** the functions above with pandas/numpy only.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add scanner indicators"`.

### Task 3: Universe and liquidity filter

**Files:**
- Create: `scanner/universe.py`, `tests/scanner/test_universe.py`

**Interfaces:**
- Produces: `normalise_ticker(t: str) -> str` (dots -> dashes, strip, upper); `fetch_sp500() -> pd.DataFrame` (columns `ticker, sector`, from Wikipedia "List of S&P 500 companies" via `pandas.read_html`, tickers normalised); `liquidity_filter(prices: dict[str, pd.DataFrame], min_dollar_vol: float, min_price: float) -> tuple[list[str], dict[str, str]]` returning eligible tickers and `{ticker: reason}` for rejects (`"price"`, `"dollar_volume"`, `"short_history"`).

- [ ] **Step 1: Write failing tests**: `test_normalise_ticker_dot_to_dash` (`"BRK.B" -> "BRK-B"`), `test_liquidity_filter_rejects_low_dollar_volume` (price 100, volume 1000 -> rejected `dollar_volume`), `test_liquidity_filter_rejects_penny_stock` (price 4.99, huge volume -> `price`), `test_liquidity_filter_rejects_short_history` (< 20 bars), `test_liquidity_filter_accepts_boundary` (exactly $50M and $5 pass).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_universe.py -v` -> FAIL.
- [ ] **Step 3: Implement.** `fetch_sp500` is not unit-tested over the network; test it only through a monkeypatched `pd.read_html` returning a small fixture table.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add universe and liquidity filter"`.

### Task 4: Data fetch adapters

**Files:**
- Create: `scanner/fetch.py`, `tests/scanner/test_fetch.py`

**Interfaces:**
- Consumes: `trader.data.download_yfinance(tickers, start)` for prices.
- Produces: `fetch_prices(tickers: list[str], start: str) -> dict[str, pd.DataFrame]`; `fetch_fundamentals(tickers) -> pd.DataFrame` indexed by ticker with columns `pe, ev_ebitda, fcf_yield, sector`; `fetch_analyst(tickers) -> pd.DataFrame` indexed by ticker with columns `rec_mean, target_gap, rec_trend` (`rec_trend` = share of buy+strongBuy now minus 3 months ago); `fetch_earnings(tickers) -> pd.Series` ticker -> next earnings `Timestamp` (NaT if unknown). All three yfinance calls return `NaN` for fields yfinance does not supply and never raise for one bad ticker (log and continue).

- [ ] **Step 1: Write failing tests** with a fake `yfinance.Ticker` injected via a `ticker_factory` parameter: `test_fundamentals_missing_field_is_nan_not_error`, `test_one_failing_ticker_does_not_abort_batch`, `test_analyst_rec_trend_computed_from_summary` (fixture counts -> expected share difference), `test_earnings_unknown_is_nat`.
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_fetch.py -v` -> FAIL.
- [ ] **Step 3: Implement** with `ticker_factory: Callable[[str], Any] = None` defaulting to a lazy `yfinance.Ticker` import; compute `fcf_yield` as free cash flow / market cap.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add yfinance fetch adapters"`.

### Task 5: Scoring, gates and selection

**Files:**
- Create: `scanner/scoring.py`, `tests/scanner/test_scoring.py`

**Interfaces:**
- Consumes: `latest_metrics` frames (Task 2), fundamentals/analyst/earnings frames (Task 4), `active_settings` (Task 1).
- Produces: `percentile_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series` (0-100, NaN stays NaN); `build_metric_table(prices, fundamentals, analyst, earnings, as_of: pd.Timestamp) -> pd.DataFrame` (one row per ticker: all metrics, `sector`, `next_earnings`); `apply_gates(table, as_of) -> tuple[pd.DataFrame, dict[str, str]]` (candidates, reject reasons: `trend`, `momentum`, `earnings_blackout`, `missing_data`); `score_table(table, settings: dict) -> pd.DataFrame` adding `technical, fundamental, sentiment, risk, blend`; `select_top(scored: pd.DataFrame, n: int = 3, max_per_sector: int = 2) -> pd.DataFrame` (sorted by `blend` desc, with a `reasons: list[str]` column of the top contributing toggles).

- [ ] **Step 1: Write failing tests**: `test_percentile_rank_ordering_and_nan`; `test_gate_rejects_below_sma200`; `test_gate_rejects_negative_63d_return`; `test_gate_earnings_blackout_five_trading_days` (earnings in 4 trading days -> rejected; in 6 -> allowed); `test_gate_missing_required_metric_reported_not_guessed`; `test_disabled_toggle_does_not_affect_score` (toggling off `rsi` leaves a score equal to the mean of the remaining technical ranks); `test_blend_uses_profile_weights` (hand-computed blend for Balanced); `test_select_top_sector_cap` (4 top names in one sector -> at most 2 chosen, next best from another sector fills); `test_select_top_returns_fewer_than_n_when_few_pass` (2 candidates -> 2 rows, no padding, no error).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_scoring.py -v` -> FAIL.
- [ ] **Step 3: Implement.** Toggle -> metric mapping: `sma_trend` close/SMA50-1 (higher better); `rsi` -abs(rsi14-55); `macd` macd_hist/close; `volume` vol_trend; `valuation_pe` -pe (pe <= 0 -> NaN); `ev_ebitda` -ev_ebitda (<= 0 -> NaN); `fcf_yield` higher better; `analyst` mean of rank(-rec_mean), rank(target_gap), rank(rec_trend). Risk score = mean of rank(-vol_20d_ann), rank(-atr_pct), rank(drawdown) (always on). A group score is the mean of its enabled toggles' ranks (NaN rank counts as 0 for that toggle). Trading-day arithmetic for the blackout uses `pandas.bdate_range`.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add scoring, gates and selection"`.

### Task 6: Sizing and risk rules

**Files:**
- Create: `scanner/sizing.py`, `tests/scanner/test_sizing.py`

**Interfaces:**
- Produces: `stop_price(entry: float, atr14: float, mult: float = 2.0) -> float`; `size_position(equity: float, entry: float, stop: float, risk_pct=0.01, cap_pct=0.20, min_order=30.0) -> dict` with keys `quantity` (float, fractional allowed), `value`, `risk_amount`, `status` in `{"ok","below_minimum","invalid"}`; `open_risk_ok(equity: float, open_risks: list[float], new_risk: float, limit_pct=0.05) -> bool`; `buys_paused(equity: float, peak_equity: float, pause_pct=0.20) -> bool`.

- [ ] **Step 1: Write failing tests**: `test_size_risks_one_percent` (equity 1000, entry 100, stop 90 -> risk_amount 10, quantity 1.0); `test_size_capped_at_twenty_percent` (tight stop would exceed cap -> value == 200); `test_size_below_minimum_reported` (value < 30 -> `below_minimum`); `test_size_invalid_when_equity_nonpositive_or_stop_above_entry` (status `invalid`, no exception); `test_open_risk_limit` (4.9% existing + 0.2% new -> False); `test_buys_paused_at_minus_twenty` (equity 800 vs peak 1000 -> True; 801 -> False).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_sizing.py -v` -> FAIL.
- [ ] **Step 3: Implement** the four functions as pure arithmetic.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add sizing and risk rules"`.

### Task 7: Paper ledger

**Files:**
- Create: `scanner/ledger.py`, `tests/scanner/test_ledger.py`

**Interfaces:**
- Consumes: `size_position`, `stop_price` (Task 6).
- Produces: `COLUMNS` list (`pick_date, ticker, config_version, status, entry_date, entry_price, stop, quantity, took_it, score_snapshot, data_timestamp, exit_date, exit_price, exit_reason, pnl, outcome`; `status` in `pending|open|closed|skipped`); `read_ledger(path) -> pd.DataFrame`; `write_ledger(df, path) -> None`; `add_picks(ledger, picks: pd.DataFrame, pick_date, version, equity) -> pd.DataFrame` (idempotent on `(pick_date, ticker, config_version)`; rows that size to `below_minimum` get status `skipped`); `fill_pending(ledger, prices, as_of) -> pd.DataFrame` (pending -> open at the first Open after `pick_date`); `process_exits(ledger, prices, as_of) -> tuple[pd.DataFrame, list[dict]]` returning updated ledger and event dicts `{ticker, kind: "stop"|"sell_signal"|"time_exit", ...}`; `mark_took_it(ledger, ticker, pick_date) -> pd.DataFrame`; `equity(ledger, start_equity, prices) -> float`; `peak_equity` is stored by the pipeline in `settings.json`; `summary_by_version(ledger) -> pd.DataFrame` (trades, wins, losses, win_rate, total_pnl per `config_version`).

- [ ] **Step 1: Write failing tests**: `test_add_picks_idempotent` (same call twice -> same row count); `test_below_minimum_pick_is_skipped_not_opened`; `test_fill_pending_uses_next_open_not_pick_day_close`; `test_stop_hit_exits_at_stop_when_bar_trades_through` (Low <= stop, Open > stop -> exit_price == stop); `test_gap_down_through_stop_exits_at_open` (Open < stop -> exit_price == Open); `test_sell_signal_closes_at_next_open`; `test_time_exit_after_56_days`; `test_outcome_win_loss_and_pnl_sign`; `test_summary_never_pools_versions` (two versions -> two rows); `test_mark_took_it_sets_flag_only`.
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_ledger.py -v` -> FAIL.
- [ ] **Step 3: Implement** with CSV storage in `scanner_state/ledger.csv`; JSON-encode `score_snapshot`; P&L = (exit - entry) x quantity.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add paper ledger"`.

### Task 8: Data archive

**Files:**
- Create: `scanner/archive.py`, `tests/scanner/test_archive.py`

**Interfaces:**
- Produces: `write_snapshot(run_date: str, frames: dict[str, pd.DataFrame], root: Path) -> list[dict]` writes `root/YYYY/MM/DD/<name>.parquet` for each named frame (`constituents, prices, fundamentals, analyst, earnings, scores`, prices concatenated with a `ticker` column) and returns manifest rows `{run_date, file, rows, sha256, fetched_at, source}`; refuses to overwrite an existing file (raises `FileExistsError`); `append_manifest(rows, manifest_path, failed: bool = False) -> None` (CSV, adds an `archive_ok` column).

- [ ] **Step 1: Write failing tests**: `test_snapshot_writes_dated_parquet_and_manifest_rows`; `test_snapshot_never_overwrites` (second call same date -> `FileExistsError`, original bytes unchanged); `test_manifest_checksum_matches_file`; `test_failed_run_marked_in_manifest` (`failed=True` -> `archive_ok` false).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_archive.py -v` -> FAIL.
- [ ] **Step 3: Implement** with `DataFrame.to_parquet` and `hashlib.sha256`.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add data archive"`.

### Task 9: Notifier and settings commands

**Files:**
- Create: `scanner/notify.py`, `tests/scanner/test_notify.py`

**Interfaces:**
- Consumes: `active_settings` (Task 1), ledger summaries (Task 7).
- Produces: `format_weekly(picks: pd.DataFrame, settings: dict, sizing: list[dict], skipped: dict[str, str]) -> str` (empty `picks` -> a message starting `No picks this week`); `format_daily(events: list[dict], open_positions: pd.DataFrame, summary: pd.DataFrame) -> str`; `split_message(text: str, limit: int = 4096) -> list[str]` (breaks on line boundaries); `class TelegramClient(token: str, chat_id: str, session=None)` with `send(text: str) -> None` (uses `split_message`) and `get_updates(offset: int | None) -> list[dict]`; `parse_command(text: str) -> dict | None` for `/profile <name>`, `/toggle <name>`, `/took <TICKER>`; `apply_commands(updates, settings, ledger) -> tuple[dict, pd.DataFrame, int]` returning new settings, new ledger and next offset; unknown commands are ignored, and commands from a chat id other than the configured one are ignored.

- [ ] **Step 1: Write failing tests**: `test_format_weekly_lists_each_pick_with_scores_stop_quantity_and_version`; `test_format_weekly_no_picks_message`; `test_split_message_respects_limit_on_line_boundaries` (5000 chars -> every part <= 4096); `test_parse_command_variants` (`/profile value`, `/toggle rsi`, `/took aapl` upper-cased, garbage -> None); `test_apply_commands_switches_profile_and_marks_custom_on_toggle`; `test_apply_commands_ignores_other_chat_ids`; `test_telegram_send_posts_each_part` (fake session records calls).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_notify.py -v` -> FAIL.
- [ ] **Step 3: Implement.** Telegram Bot API over `requests` (`sendMessage`, `getUpdates` with `offset`); inline buttons are not used in v1: commands only (spec allows "buttons / commands"). No token appears in any log line or exception message.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add notifier and settings commands"`.

### Task 10: Google Drive upload and Sheets mirror

**Files:**
- Create: `scanner/google_io.py`, `tests/scanner/test_google_io.py`

**Interfaces:**
- Produces: `build_services(service_account_json: str) -> tuple[Any, Any]` (Drive v3, Sheets v4); `upload_to_drive(drive, local_root: Path, rel_paths: list[str], folder_id: str) -> None` (creates the dated subfolders under `folder_id`, skips files already present by name, never deletes); `mirror_to_sheets(sheets, sheet_id: str, tabs: dict[str, pd.DataFrame]) -> None` (clears and rewrites only the listed tabs, e.g. `Picks, Open, WinLoss, Scores`; never touches other tabs); `check_drive_write(drive, folder_id: str) -> str` (uploads and deletes a tiny `_sa_write_test.txt`, returns its file id or raises with a message naming quota/permission).

- [ ] **Step 1: Write failing tests** with fake Drive/Sheets service objects: `test_upload_skips_existing_names`; `test_upload_creates_dated_subfolders`; `test_mirror_only_touches_listed_tabs` (a "Notes" tab in the fake sheet is untouched); `test_check_drive_write_raises_on_quota_error` (fake raises 403 storageQuotaExceeded -> message mentions quota).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_google_io.py -v` -> FAIL.
- [ ] **Step 3: Implement** with `google.oauth2.service_account.Credentials.from_service_account_info` and scopes `drive` and `spreadsheets`.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add Drive upload and Sheets mirror"`.

### Task 11: Pipeline, CLI and offline dry run

**Files:**
- Create: `scanner/pipeline.py`, `scanner/cli.py`, `tests/scanner/test_pipeline.py`
- Modify: `scanner/__main__.py` (call `cli.main()`)

**Interfaces:**
- Consumes: every module above.
- Produces: `run_weekly(cfg, deps, as_of: pd.Timestamp, state_dir: Path) -> dict` and `run_daily(...) -> dict`, where `deps` is a dataclass `Deps(fetch_prices, fetch_fundamentals, fetch_analyst, fetch_earnings, fetch_sp500, telegram, drive_upload, sheets_mirror)` so tests inject fakes; result dict has `picks`, `events`, `archive_ok`, `errors`. Weekly order: apply Telegram commands -> universe + liquidity -> metrics -> gates -> score -> select -> size -> ledger -> archive -> Telegram -> Drive -> Sheets. Daily order: apply commands -> fetch prices for open/pending tickers -> fill pending -> process exits -> update peak equity -> Telegram -> archive prices -> Sheets. CLI: `python -m scanner weekly|daily|check-drive [--as-of YYYY-MM-DD]`; environment variables supply secrets.

- [ ] **Step 1: Write failing tests** using `synthetic_prices` from `trader.data` and fake deps: `test_weekly_end_to_end_offline` (picks <= 3, ledger rows pending, one Telegram message, archive files exist); `test_weekly_rerun_same_week_no_duplicates_and_no_second_message`; `test_weekly_sends_no_picks_message_when_nothing_passes`; `test_archive_failure_alerts_but_telegram_and_ledger_still_done` (fake `drive_upload` raises -> result `archive_ok` False, ledger written, Telegram sent, manifest marks failure); `test_sheets_failure_does_not_block`; `test_buys_paused_when_drawdown_exceeds_limit_still_sends_alerts`; `test_daily_fills_pending_then_exits_on_stop`.
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_pipeline.py -v` -> FAIL.
- [ ] **Step 3: Implement** the two runners and CLI; each side-effecting step is wrapped so its failure is recorded in `errors` and an alert is sent, then the run continues (except a failure to read/write the ledger, which aborts with a Telegram alert).
- [ ] **Step 4: Run** the full suite `uv run pytest -q` -> all scanner and existing tests PASS.
- [ ] **Step 5: Commit** `git commit -m "Add scanner pipeline and CLI"`.

### Task 12: GitHub Actions workflows and setup guide

**Files:**
- Create: `.github/workflows/scanner-weekly.yml`, `.github/workflows/scanner-daily.yml`, `scanner/README.md`, `tests/scanner/test_workflows.py`

**Interfaces:**
- Consumes: CLI from Task 11; secrets listed in Global Constraints.

- [ ] **Step 1: Write failing test** `test_workflows_parse_and_reference_cli`: both YAML files load, define a `schedule` cron and `workflow_dispatch`, use `uv run python -m scanner weekly|daily`, grant `contents: write`, and mention no literal secret values (secrets only as `${{ secrets.* }}`).
- [ ] **Step 2: Run** `uv run pytest tests/scanner/test_workflows.py -v` -> FAIL.
- [ ] **Step 3: Create the workflows.** Weekly: Sunday 22:00 UTC, so the next Monday US open fills the pick; daily: 21:30 UTC Monday-Friday, after the US close. Both: checkout, setup uv, `uv sync`, run CLI with secrets as env, then commit `scanner_state/` back (`git add scanner_state && git diff --cached --quiet || git commit -m "scanner: update state" && git push`). Write `scanner/README.md` with the setup steps: create the Telegram bot and chat id, create the Google Cloud service account, enable Drive and Sheets APIs, share the Drive folder (`1WMlMlhfo9qGcnoAWESNs3WIp6bhv6qXH`) and the sheet with the service account email as editor, add the five secrets, run `check-drive` once, then trigger the weekly workflow manually.
- [ ] **Step 4: Run** `uv run pytest -q` -> PASS.
- [ ] **Step 5: Commit** `git commit -m "Add scanner workflows and setup guide"`.

---

## Self-review

- **Spec coverage:** universe + liquidity (T3), blend/gates/sector cap/earnings blackout (T5), sizing/stop/caps/pause (T6), exits/entry/paper account (T7), ledger + versioned results (T7), archive (T8), Telegram + profiles/toggles/took-it (T9), Drive + Sheets mirror (T10), cadence, failure isolation and idempotency (T11), hosting (T12). Spec "no backtest", "live ordering", "news sentiment" are intentionally absent.
- **Type consistency:** `config_version`/`active_settings` (T1) are consumed in T5, T7, T9, T11; `size_position` returns `status` strings that T7 and T9 use verbatim (`below_minimum`).
- **Proportion:** tasks list signatures, test names and pinned values only; no function bodies.
- **Known open values (proposed in the spec, user may change in `scanner.yaml`):** liquidity thresholds, 5-day blackout, 2x ATR, `start_equity: 1000`, cron times.
