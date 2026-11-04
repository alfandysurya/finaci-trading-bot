"""Weekly and daily scanner runs: wire the modules together. PAPER ONLY.

Nothing here places or cancels a broker order. Every side effect after the ledger is
saved (archive, Telegram, Drive, Sheets) is isolated: its failure is recorded in
`errors`, alerted on Telegram and the run continues. A failure to read or write the
ledger (or the other state files) aborts the run after an alert.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from scanner import ledger as L
from scanner.archive import append_manifest, write_snapshot
from scanner.config import ROOT, active_settings
from scanner.fetch import ANALYST_COLS, FUNDAMENTAL_COLS
from scanner.notify import (FOOTER, apply_commands, format_daily, format_summary,
                            format_weekly)
from scanner.scoring import apply_gates, build_metric_table, score_table, select_top
from scanner.sizing import buys_paused, open_risk_ok, size_position, stop_price
from scanner.universe import liquidity_filter

ENV_TOKEN = "TELEGRAM_BOT_TOKEN"
ENV_CHAT = "TELEGRAM_CHAT_ID"
ENV_SA = "GOOGLE_SERVICE_ACCOUNT_JSON"
ENV_DRIVE = "DRIVE_FOLDER_ID"
ENV_SHEET = "SHEET_ID"
SECRET_ENV = (ENV_TOKEN, ENV_CHAT, ENV_SA, ENV_DRIVE, ENV_SHEET)

PRICE_CHUNK = 100
LOOKBACK_DAYS = 400
TOP_N = 3
UNDELIVERED_CAP = 20      # stored undelivered messages (oldest dropped)
MIN_PRICE_COVERAGE = 0.5  # weekly aborts below this share of constituents with prices
DATA_WARN_PCT = 30        # warn when more than this % of stage-2 rows lack a data group
EARNINGS_ABORT_PCT = 70   # weekly aborts when more than this % of earnings dates are unknown
FRESH_DAYS = 4            # daily: a held ticker's last bar must be within as_of - 4 days
_MIN_SECRET_LEN = 4  # shorter values would mangle ordinary text when scrubbed


@dataclass
class Deps:
    fetch_prices: Callable[[list[str], str], dict[str, pd.DataFrame]]
    fetch_fundamentals: Callable[[list[str]], pd.DataFrame]
    fetch_analyst: Callable[[list[str]], pd.DataFrame]
    fetch_earnings: Callable[[list[str]], pd.Series]
    fetch_sp500: Callable[[], pd.DataFrame]
    telegram: Any
    drive_upload: Callable[[Path, list[str]], None] | None = None
    sheets_mirror: Callable[[dict[str, pd.DataFrame]], None] | None = None
    chat_id: str = "0"
    secrets: tuple[str, ...] = field(default=(), repr=False)
    dry_run: bool = False


# ------------------------------------------------------------ secrets / adapters
def secret_values(env: Mapping[str, str]) -> tuple[str, ...]:
    """Every secret value worth scrubbing from error text (env values plus key material)."""
    vals = [env.get(k) or "" for k in SECRET_ENV]
    try:
        info = json.loads(env.get(ENV_SA) or "")
        if isinstance(info, dict):
            vals += [str(info.get(k) or "") for k in ("private_key", "private_key_id")]
    except ValueError:
        pass
    return tuple(v for v in vals if len(v) >= _MIN_SECRET_LEN)


def scrub(text: str, secrets) -> str:
    for s in sorted(secrets, key=len, reverse=True):
        if s and len(s) >= _MIN_SECRET_LEN:
            text = text.replace(s, "***")
    return text


class ConsoleNotifier:
    """Dry-run stand-in for TelegramClient: prints messages, never receives commands."""

    def send(self, text: str) -> None:
        print(text)
        print("-" * 40)

    def get_updates(self, offset) -> list[dict]:
        return []


class _Google:
    """Builds the Google clients on first use, so bad credentials fail inside the
    isolated Drive/Sheets step instead of before the ledger is touched."""

    def __init__(self, service_account_json: str):
        self._sa = service_account_json
        self._services = None

    def __repr__(self) -> str:
        return "_Google()"

    def services(self):
        if self._services is None:
            from scanner.google_io import build_services

            self._services = build_services(self._sa)
        return self._services


class _DriveUpload:
    def __init__(self, google: _Google, folder_id: str):
        self._google, self._folder = google, folder_id

    def __repr__(self) -> str:
        return "DriveUpload()"

    def __call__(self, local_root: Path, rel_paths: list[str]) -> None:
        from scanner.google_io import upload_to_drive

        upload_to_drive(self._google.services()[0], local_root, rel_paths, self._folder)


class _SheetsMirror:
    def __init__(self, google: _Google, sheet_id: str):
        self._google, self._sheet = google, sheet_id

    def __repr__(self) -> str:
        return "SheetsMirror()"

    def __call__(self, tabs: dict[str, pd.DataFrame]) -> None:
        from scanner.google_io import mirror_to_sheets

        mirror_to_sheets(self._google.services()[1], self._sheet, tabs)


def default_deps(env: Mapping[str, str], dry_run: bool) -> Deps:
    """Real adapters from environment variables. Values are never printed."""
    from scanner import fetch, universe
    from scanner.notify import TelegramClient

    # One rate-limit state per run (one run per process): once a fetch's breaker trips,
    # the later fetches skip the long retry waits (see scanner.fetch).
    rate = fetch.RateLimitState()
    common = dict(fetch_prices=fetch.fetch_prices,
                  fetch_fundamentals=partial(fetch.fetch_fundamentals, rate_state=rate),
                  fetch_analyst=partial(fetch.fetch_analyst, rate_state=rate),
                  fetch_earnings=partial(fetch.fetch_earnings, rate_state=rate),
                  fetch_sp500=universe.fetch_sp500, secrets=secret_values(env))
    if dry_run:
        return Deps(telegram=ConsoleNotifier(), drive_upload=None, sheets_mirror=None,
                    chat_id="0", dry_run=True, **common)
    missing = [k for k in (ENV_TOKEN, ENV_CHAT) if not env.get(k)]
    if missing:
        raise SystemExit("Missing required environment variable(s): " + ", ".join(missing)
                         + ". Set them, or use --dry-run.")
    chat_id = str(env[ENV_CHAT])
    google = _Google(env[ENV_SA]) if env.get(ENV_SA) else None
    drive = _DriveUpload(google, env[ENV_DRIVE]) if google and env.get(ENV_DRIVE) else None
    sheets = _SheetsMirror(google, env[ENV_SHEET]) if google and env.get(ENV_SHEET) else None
    return Deps(telegram=TelegramClient(env[ENV_TOKEN], chat_id), drive_upload=drive,
                sheets_mirror=sheets, chat_id=chat_id, **common)


# ------------------------------------------------------------ run context
class _Abort(RuntimeError):
    pass


class _Run:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.errors: list[str] = []
        self.sent: list[str] = []
        self.step = "startup"  # label for the catch-all alert in the runners

    def describe(self, exc: BaseException) -> str:
        return scrub(f"{type(exc).__name__}: {exc}", self.deps.secrets)

    def send(self, text: str) -> bool:
        try:
            self.deps.telegram.send(text)
        except Exception as exc:  # never alert about Telegram via Telegram
            self.errors.append(f"telegram: {self.describe(exc)}")
            return False
        self.sent.append(text)
        return True

    def fail(self, step: str, exc: BaseException) -> None:
        msg = self.describe(exc)
        self.errors.append(f"{step}: {msg}")
        self.send(f"ALERT: {step} failed - {msg}")

    def critical(self, step: str, exc: BaseException):
        """Alert, then abort the run (state files are the source of truth)."""
        self.fail(step, exc)
        raise _Abort(f"{step} failed") from exc


# ------------------------------------------------------------ state files
def _paths(state_dir: Path) -> dict[str, Path]:
    return {name: state_dir / f for name, f in [
        ("ledger", "ledger.csv"), ("settings", "settings.json"), ("state", "state.json"),
        ("offset", "telegram_offset.json"), ("manifest", "manifest.csv")]}


def _replace_into(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _write_json(path: Path, data: dict) -> None:
    _replace_into(path, lambda p: p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8"))


def _read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} does not hold a JSON object")
    return {**default, **data}


def _save_ledger(run: _Run, ledger: pd.DataFrame, path: Path) -> None:
    try:
        _replace_into(path, lambda p: L.write_ledger(ledger, p))
    except Exception as exc:
        run.critical("ledger write", exc)


def _save_json(run: _Run, path: Path, data: dict) -> None:
    try:
        _write_json(path, data)
    except Exception as exc:
        run.critical(f"{path.name} write", exc)


def _load(run: _Run, paths: dict[str, Path]):
    try:
        ledger = L.read_ledger(paths["ledger"])
    except Exception as exc:
        run.critical("ledger read", exc)
    try:
        settings = _read_json(paths["settings"], {"profile": "balanced"})
        state = _read_json(paths["state"], {"peak_equity": None, "last_weekly_pick_date": None,
                                            "undelivered": []})
        if not isinstance(state["undelivered"], list) or not all(
                isinstance(t, str) for t in state["undelivered"]):
            raise ValueError("state.json 'undelivered' must be a list of strings")
    except Exception as exc:
        run.critical("state read", exc)
    return ledger, settings, state


def _commands(run: _Run, cfg: dict, paths: dict[str, Path], ledger: pd.DataFrame,
              settings: dict):
    """Apply queued Telegram commands at most once.

    The new offset is persisted FIRST, as soon as the commands are applied in memory,
    so a crash afterwards can lose a command but never replay it (a replayed /toggle
    would undo itself). Then ledger and settings are saved, then replies are sent.
    A missing offset file means first run (offset None); an unreadable one aborts."""
    run.step = "telegram commands"
    try:
        offset = _read_json(paths["offset"], {"offset": None}).get("offset")
        if offset is not None and (isinstance(offset, bool) or not isinstance(offset, int)):
            raise ValueError("telegram_offset.json holds a non-integer offset")
    except Exception as exc:
        run.critical("telegram offset read", exc)
    try:
        updates = run.deps.telegram.get_updates(offset)
    except Exception as exc:
        run.fail("telegram updates", exc)
        updates = []
    # No filtering by id: Telegram already honours the offset, and after a quiet week
    # it may restart update_id at a random (possibly lower) value. Only drop updates
    # whose update_id is missing or not an integer, which apply_commands cannot order.
    updates = [u for u in updates if isinstance(u, dict)
               and isinstance(u.get("update_id"), int)
               and not isinstance(u.get("update_id"), bool)]
    settings, ledger, next_offset, replies = apply_commands(
        updates, settings, ledger, cfg, run.deps.chat_id)
    if next_offset is not None:  # no updates -> keep the stored offset
        try:
            _write_json(paths["offset"], {"offset": int(next_offset)})
        except Exception as exc:
            run.critical("telegram offset write", exc)
    _save_ledger(run, ledger, paths["ledger"])
    _save_json(run, paths["settings"], settings)
    for reply in replies:
        run.send(reply)
    return settings, ledger


# ------------------------------------------------------------ helpers
def _day(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _slice(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Drop every bar after as_of (look-ahead guard) and frames left with no bars."""
    out = {}
    for t, df in prices.items():
        if df is None or df.empty:
            continue
        cut = df[df.index <= as_of]
        if not cut.empty:
            out[t] = cut
    return out


def _fetch_prices(deps: Deps, tickers: list[str], as_of: pd.Timestamp) -> dict:
    start = (as_of - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    raw: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), PRICE_CHUNK):
        raw.update(deps.fetch_prices(tickers[i:i + PRICE_CHUNK], start) or {})
    return _slice(raw, as_of)


def _active_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    return ledger[ledger["status"].isin(["open", "pending"])]


def _mismatched(row: pd.Series, prices: dict) -> bool:
    """Same test as scanner.ledger.process_exits: the entry-date bar's Open must exist
    and lie within ledger.MISMATCH_TOL of the stored entry_price. Rows without price
    data are not mismatches (process_exits skips them; equity() values them at 0)."""
    df = prices.get(row["ticker"])
    if df is None or df.empty:
        return False
    ts = pd.Timestamp(row["entry_date"])
    if ts not in df.index:
        return True
    observed = df.loc[ts, "Open"]
    if isinstance(observed, pd.Series):
        observed = observed.iloc[-1]
    if pd.isna(observed):
        return True
    return abs(float(observed) / float(row["entry_price"]) - 1) > L.MISMATCH_TOL


def _unvalued(row: pd.Series, prices: dict) -> bool:
    """An open row that cannot be valued: no price frame, or a data mismatch."""
    df = prices.get(row["ticker"])
    return df is None or df.empty or _mismatched(row, prices)


def _valued_equity(ledger: pd.DataFrame, prices: dict, start_equity: float) -> float:
    """ledger.equity() without open rows that cannot be valued: prices re-adjusted
    since entry (split/dividend) or no price frame at all. Used by both runners."""
    is_open = ledger["status"] == "open"
    bad = pd.Series([bool(is_open.iloc[i]) and _unvalued(ledger.iloc[i], prices)
                     for i in range(len(ledger))], index=ledger.index, dtype=bool)
    return L.equity(ledger[~bad], float(start_equity), prices)


def _position_values(open_rows: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """last_close / unrealised / unrealised_pct per valuable open row (same index)."""
    recs = {}
    for idx, row in open_rows.iterrows():
        if _unvalued(row, prices):
            continue
        closes = prices[row["ticker"]]["Close"].dropna()
        entry, qty = _float(row["entry_price"], math.nan), _float(row["quantity"], math.nan)
        if closes.empty or not (math.isfinite(entry) and entry > 0 and math.isfinite(qty)):
            continue
        last = float(closes.iloc[-1])
        recs[idx] = {"last_close": last, "unrealised": qty * (last - entry),
                     "unrealised_pct": (last / entry - 1) * 100}
    return pd.DataFrame.from_dict(recs, orient="index",
                                  columns=["last_close", "unrealised", "unrealised_pct"])


def _row_risk(row: pd.Series, prices: dict) -> float:
    """Risk still at stake for an open/pending row: quantity * (entry - stop)."""
    qty, stop = _float(row["quantity"], math.nan), _float(row["stop"], math.nan)
    entry = _float(row["entry_price"], math.nan)
    if not math.isfinite(entry):  # pending: sized against the pick's reference close
        try:
            entry = _float(json.loads(row["score_snapshot"] or "{}").get("close"), math.nan)
        except ValueError:
            entry = math.nan
    if not math.isfinite(entry):
        df = prices.get(row["ticker"])
        entry = _float(df["Close"].dropna().iloc[-1], math.nan) if df is not None else math.nan
    if not (math.isfinite(qty) and math.isfinite(stop) and math.isfinite(entry)):
        return 0.0
    return max(0.0, qty * (entry - stop))


def _mirror_tabs(ledger: pd.DataFrame, scores: pd.DataFrame | None = None) -> dict:
    tabs = {"Picks": ledger.reset_index(drop=True),
            "Open": _active_rows(ledger).reset_index(drop=True),
            "WinLoss": L.summary_by_version(ledger)}
    if scores is not None:
        tabs["Scores"] = scores
    return tabs


def _mirror(run: _Run, tabs: dict) -> None:
    if run.deps.sheets_mirror is None:
        return
    try:
        run.deps.sheets_mirror(tabs)
    except Exception as exc:
        run.fail("sheets mirror", exc)


def _write_archive(run: _Run, day: str, frames: dict, root: Path) -> list[dict] | None:
    try:
        return write_snapshot(day, frames, root)
    except Exception as exc:
        run.fail("archive", exc)
        return None


def _upload_archive(run: _Run, day: str, rows: list[dict] | None, root: Path,
                    manifest: Path) -> bool:
    """Upload written files to Drive, then record the manifest (failed rows on error).

    Outside a dry run, a missing Drive configuration leaves the archive only on the
    runner's disk (lost on GitHub Actions): the local manifest rows are kept as
    written, but the result reports archive_ok False and one alert is sent."""
    ok = rows is not None
    not_uploaded = False
    if rows is not None and run.deps.drive_upload is not None:
        run.step = "drive upload"
        try:
            run.deps.drive_upload(root, [r["file"] for r in rows])
        except Exception as exc:
            run.fail("drive upload", exc)
            ok = False
    elif rows is not None and not run.deps.dry_run:
        not_uploaded = True
        ok = False  # never uploaded: the manifest must say so
    try:
        append_manifest(rows or [], manifest, failed=not ok, run_date=day)
    except Exception as exc:
        run.fail("manifest", exc)
    if not_uploaded:
        msg = "Drive not configured; archive not uploaded"
        run.errors.append(msg)
        run.send(f"ALERT: drive upload failed - {msg}")
    return ok


def _archive_root(cfg: dict) -> Path:
    return ROOT / cfg["archive"]["archive_dir"]


def _prices_frame(prices: dict[str, pd.DataFrame], tickers) -> pd.DataFrame:
    parts = []
    for t in tickers:
        df = prices[t].copy()
        df.index = pd.DatetimeIndex(df.index, name="date")
        parts.append(df.reset_index().assign(ticker=t))
    if not parts:
        return pd.DataFrame(columns=["date", "Open", "High", "Low", "Close", "Volume", "ticker"])
    return pd.concat(parts, ignore_index=True)


def _join_reasons(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.reset_index()
    if "reasons" in out.columns:
        out["reasons"] = out["reasons"].map(
            lambda v: "; ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
    return out


def _result(run: _Run) -> dict:
    return {"picks": pd.DataFrame(), "events": [], "archive_ok": True, "errors": run.errors,
            "sent": run.sent, "skipped_reason": None, "send_failed": False}


def _store_undelivered(run: _Run, state: dict, paths: dict[str, Path], texts: list[str]) -> None:
    state["undelivered"] = list(texts)[-UNDELIVERED_CAP:]  # drop oldest beyond the cap
    _save_json(run, paths["state"], state)


def _redeliver(run: _Run, state: dict, paths: dict[str, Path], res: dict) -> None:
    """Re-send stored main messages (oldest first); keep only those that fail again."""
    pending = list(state.get("undelivered") or [])
    if not pending:
        return
    run.step = "redelivery"
    left = [text for text in pending if not run.send(text)]
    if left:
        res["send_failed"] = True
    _store_undelivered(run, state, paths, left)


def _send_main(run: _Run, state: dict, paths: dict[str, Path], res: dict, text: str) -> None:
    """Send a weekly/daily/paused message; on failure keep it for the next run and
    mark the run so the CLI exits non-zero (after all state is saved)."""
    if run.send(text):
        return
    res["send_failed"] = True
    _store_undelivered(run, state, paths, list(state.get("undelivered") or []) + [text])


def _gics_fundamentals(fund: pd.DataFrame, tickers: list[str], gics: dict) -> pd.DataFrame:
    """Fundamentals for `tickers` with the GICS sector from the constituent list: one
    sector source for every ticker, so the per-sector cap never counts one sector under
    two different labels (and stage 1 has a sector before Yahoo is asked)."""
    out = fund.reindex(tickers, columns=FUNDAMENTAL_COLS).copy()
    out["sector"] = [gics.get(t) for t in out.index]
    return out


def _price_survivors(prices: dict, eligible: list[str], gics: dict, held: list[str],
                     as_of: pd.Timestamp, gates: dict, rejects: dict) -> list[str]:
    """Weekly stage 1: gates on a price-only metric table (fundamentals, analyst and
    earnings all empty, so the earnings blackout cannot fire yet), minus held tickers.
    Gate rejects are recorded in `rejects`; survivors keep the `eligible` order."""
    if not eligible:
        return []
    fund = _gics_fundamentals(pd.DataFrame(columns=FUNDAMENTAL_COLS), eligible, gics)
    table = build_metric_table({t: prices[t] for t in eligible}, fund,
                               pd.DataFrame(columns=ANALYST_COLS),
                               pd.Series(dtype="datetime64[ns]"), as_of)
    candidates, gate_rejects = apply_gates(table, as_of, gates["stale_days"],
                                           gates["earnings_blackout_days"])
    rejects.update(gate_rejects)
    # Never re-pick a ticker that is already open or pending.
    return [t for t in candidates.index if t not in set(held)]


def _unknown_earnings(earnings: pd.Series, tickers: list[str]) -> int:
    return int(pd.to_datetime(earnings.reindex(tickers), errors="coerce").isna().sum())


def _data_warning(fund: pd.DataFrame, analyst: pd.DataFrame, earnings: pd.Series,
                  tickers: list[str]) -> str | None:
    """Coverage over the stage-2 tickers (the only ones fetched from Yahoo)."""
    m = len(tickers)
    if not m:
        return None
    num_fund = [c for c in FUNDAMENTAL_COLS if c != "sector"]
    n = int(fund.reindex(tickers, columns=num_fund).isna().all(axis=1).sum())
    k = int(analyst.reindex(tickers, columns=ANALYST_COLS).isna().all(axis=1).sum())
    e = _unknown_earnings(earnings, tickers)
    parts = []
    if n * 100 > DATA_WARN_PCT * m or k * 100 > DATA_WARN_PCT * m:
        parts.append(f"fundamentals missing for {n}/{m}, analyst data missing for "
                     f"{k}/{m} - scores for those groups are unreliable")
    if e * 100 > DATA_WARN_PCT * m:
        parts.append(f"earnings dates unknown for {e}/{m} - earnings blackout not "
                     "applied to them")
    return "DATA WARNING: " + "; ".join(parts) if parts else None


def _with_warning(text: str, warning: str | None) -> str:
    """Insert the warning line just above the footer."""
    if not warning:
        return text
    lines = text.split("\n")
    return "\n".join(lines[:-1] + [warning, ""] + lines[-1:])


def _guarded(body, run: _Run, *args) -> dict:
    """Run a runner body. A state abort becomes RuntimeError (already alerted); any
    other unexpected exception is alerted (scrubbed) under the current step and re-raised."""
    try:
        return body(run, *args)
    except _Abort:
        raise RuntimeError("; ".join(run.errors)) from None
    except Exception as exc:
        run.fail(run.step, exc)
        raise


# ------------------------------------------------------------ weekly
def run_weekly(cfg: dict, deps: Deps, as_of: pd.Timestamp, state_dir: Path,
               archive_root: Path | None = None) -> dict:
    root = Path(archive_root) if archive_root is not None else _archive_root(cfg)
    return _guarded(_weekly, _Run(deps), cfg, _day(as_of), Path(state_dir), root)


def _weekly(run: _Run, cfg: dict, as_of: pd.Timestamp, state_dir: Path,
            root: Path) -> dict:
    res = _result(run)
    paths = _paths(state_dir)
    day = as_of.strftime("%Y-%m-%d")
    run.step = "state load"
    ledger, settings, state = _load(run, paths)
    settings, ledger = _commands(run, cfg, paths, ledger, settings)
    _redeliver(run, state, paths, res)
    run.step = "weekly run"

    if (ledger["pick_date"] == day).any() or state.get("last_weekly_pick_date") == day:
        res["skipped_reason"] = f"weekly already ran for {day}"
        return res

    active = active_settings(cfg, settings)
    uni, gates, sz = cfg["universe"], cfg["gates"], cfg["sizing"]
    start_equity = float(cfg["paper_account"]["start_equity"])
    frames: dict[str, pd.DataFrame] = {}
    scores_tab = pd.DataFrame(columns=["ticker"])
    try:
        constituents = run.deps.fetch_sp500()
        frames["constituents"] = constituents
        gics = dict(zip(constituents["ticker"], constituents["sector"]))
        held = list(_active_rows(ledger)["ticker"])
        tickers = list(dict.fromkeys(list(constituents["ticker"]) + held))
        prices = _fetch_prices(run.deps, tickers, as_of)
        uni_prices = {t: prices[t] for t in constituents["ticker"] if t in prices}
        total = len(constituents)
        # "Usable" = a frame whose last bar would pass the stale gate; frames that stop
        # days early would all be rejected as stale_data and look like "No picks".
        fresh_from = as_of - pd.Timedelta(days=gates["stale_days"])
        usable = sum(1 for df in uni_prices.values() if _day(df.index[-1]) >= fresh_from)
        if total == 0 or usable < MIN_PRICE_COVERAGE * total:
            # A data outage, not a quiet week: alert as a scan failure (no marker set).
            raise RuntimeError(f"price data covers {usable} of {total} constituents")
        eligible, rejects = liquidity_filter(uni_prices, uni["min_dollar_volume"],
                                             uni["min_price"], uni.get("window", 20))
        for t in constituents["ticker"]:
            if t not in prices:
                rejects[t] = "no_data"
        frames["prices"] = _prices_frame(prices, eligible)

        equity = _valued_equity(ledger, prices, start_equity)
        peak = _float(state.get("peak_equity"))
        paused = buys_paused(equity, peak, sz["pause_drawdown_pct"])
        picks, sizing, stops, skipped, to_ledger = pd.DataFrame(), [], [], {}, []
        warning = None
        if not paused:
            # Stage 1: price-only gates (stale, missing_data, trend, momentum) on the
            # liquid universe, so the slow per-ticker Yahoo calls below are made only
            # for the stocks that can still be picked.
            survivors = _price_survivors(prices, eligible, gics, held, as_of, gates, rejects)
            # Stage 2: earnings, fundamentals and analyst for the survivors only. The
            # archived fundamentals/analyst/earnings frames therefore cover these
            # stage-2 tickers, not the whole liquid universe. Earnings come first, so a
            # blind blackout aborts after one slow fetch, not three.
            earnings = (run.deps.fetch_earnings(survivors) if survivors
                        else pd.Series(dtype="datetime64[ns]"))
            unknown = _unknown_earnings(earnings, survivors)
            if unknown * 100 > EARNINGS_ABORT_PCT * len(survivors):
                # A blind earnings blackout must not send picks as normal: alert as a
                # scan failure (no marker, no ledger rows), so a rerun later is allowed.
                raise RuntimeError(f"earnings dates unknown for {unknown}/{len(survivors)} "
                                   "stocks - earnings blackout cannot be checked")
            fund = (run.deps.fetch_fundamentals(survivors) if survivors
                    else pd.DataFrame(columns=FUNDAMENTAL_COLS))
            analyst = (run.deps.fetch_analyst(survivors) if survivors
                       else pd.DataFrame(columns=ANALYST_COLS))
            warning = _data_warning(fund, analyst, earnings, survivors)
            if warning:
                run.errors.append(warning)
            fund = _gics_fundamentals(fund, survivors, gics)
            frames["fundamentals"] = fund.rename_axis("ticker")
            frames["analyst"] = analyst.reindex(survivors).rename_axis("ticker")
            frames["earnings"] = (earnings.reindex(survivors).rename("next_earnings")
                                  .rename_axis("ticker").to_frame())
            scored = None
            if survivors:
                table = build_metric_table({t: prices[t] for t in survivors}, fund, analyst,
                                           earnings, as_of)
                # Same gates again: now the earnings blackout can apply.
                candidates, gate_rejects = apply_gates(
                    table, as_of, gates["stale_days"], gates["earnings_blackout_days"])
                rejects.update(gate_rejects)
                if len(candidates):
                    scored = score_table(candidates, active)
            if scored is not None:
                ranked = select_top(scored, n=len(scored), max_per_sector=len(scored))
                scores_tab = _join_reasons(ranked)
                # max_per_sector counts only this week's picks, not positions already held.
                top = select_top(scored, n=TOP_N, max_per_sector=gates["max_per_sector"])
                open_risks = [_row_risk(r, prices) for _, r in _active_rows(ledger).iterrows()]
                accepted = []
                for t, row in top.iterrows():
                    close, atr = float(row["close"]), float(row["atr14"])
                    stop = stop_price(close, atr, sz["stop_atr_mult"])
                    size = size_position(equity, close, stop, sz["risk_pct"], sz["cap_pct"],
                                         sz["min_order"])
                    if size["status"] != "ok":
                        skipped[t] = size["status"].replace("_", " ")
                        to_ledger.append(t)  # recorded by the ledger as a skipped row
                        continue
                    if not open_risk_ok(equity, open_risks, size["risk_amount"],
                                        sz["open_risk_limit_pct"]):
                        skipped[t] = "open risk limit reached"
                        continue
                    open_risks.append(size["risk_amount"])
                    accepted.append(t)
                    to_ledger.append(t)
                    sizing.append(size)
                    stops.append(stop)
                top["data_timestamp"] = top["last_bar"].map(
                    lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
                picks = top.loc[accepted]
                if to_ledger:
                    ledger = L.add_picks(ledger, top.loc[to_ledger], day, active["version"],
                                         equity, stop_mult=sz["stop_atr_mult"],
                                         risk_pct=sz["risk_pct"], cap_pct=sz["cap_pct"],
                                         min_order=sz["min_order"])
        for t in held:  # every held ticker is reported as such, whatever else it failed
            rejects[t] = "already held"
        frames["scores"] = scores_tab
        frames["rejects"] = pd.DataFrame({"ticker": list(rejects),
                                          "reason": list(rejects.values())},
                                         columns=["ticker", "reason"])
    except Exception as exc:
        run.fail("scan", exc)
        res["archive_ok"] = False
        try:
            append_manifest([], paths["manifest"], failed=True, run_date=day)
        except Exception as mexc:
            run.fail("manifest", mexc)
        # No Sheets mirror here: `ledger` may hold picks that were never saved.
        return res

    _save_ledger(run, ledger, paths["ledger"])
    state["last_weekly_pick_date"] = day
    _save_json(run, paths["state"], state)
    res["picks"] = picks

    if paused:
        frames = {k: frames[k] for k in ("constituents", "prices")}
    run.step = "archive"
    rows = _write_archive(run, day, frames, root)

    run.step = "weekly message"
    if paused:
        drop = (1 - equity / peak) * 100
        text = f"Buys paused: equity is {drop:.1f}% below peak\n{FOOTER}"
    else:
        text = _with_warning(format_weekly(picks, active, sizing, skipped, stops), warning)
    _send_main(run, state, paths, res, text)

    res["archive_ok"] = _upload_archive(run, day, rows, root, paths["manifest"])
    run.step = "sheets mirror"
    _mirror(run, _mirror_tabs(ledger, scores_tab))
    return res


# ------------------------------------------------------------ daily
def run_daily(cfg: dict, deps: Deps, as_of: pd.Timestamp, state_dir: Path,
              archive_root: Path | None = None) -> dict:
    root = Path(archive_root) if archive_root is not None else _archive_root(cfg)
    return _guarded(_daily, _Run(deps), cfg, _day(as_of), Path(state_dir), root)


def _daily(run: _Run, cfg: dict, as_of: pd.Timestamp, state_dir: Path, root: Path) -> dict:
    res = _result(run)
    paths = _paths(state_dir)
    day = as_of.strftime("%Y-%m-%d")
    run.step = "state load"
    ledger, settings, state = _load(run, paths)
    settings, ledger = _commands(run, cfg, paths, ledger, settings)
    _redeliver(run, state, paths, res)
    run.step = "daily positions"

    tickers = list(dict.fromkeys(_active_rows(ledger)["ticker"]))
    prices: dict[str, pd.DataFrame] = {}
    if tickers:
        try:
            prices = _fetch_prices(run.deps, tickers, as_of)
        except Exception as exc:
            # The daily message itself becomes the alert (below); peak stays unchanged.
            run.errors.append(f"fetch prices: {run.describe(exc)}")
            run.step = "daily message"
            lines = ["ALERT: prices unavailable - stops not checked today"]
            summary = format_summary(L.summary_by_version(ledger))
            if summary:
                lines += [""] + summary
            _send_main(run, state, paths, res, "\n".join(lines))
            run.step = "sheets mirror"
            _mirror(run, _mirror_tabs(ledger))
            return res

    cutoff = as_of - pd.Timedelta(days=FRESH_DAYS)
    unchecked = [t for t in tickers if t not in prices or _day(prices[t].index[-1]) < cutoff]
    if unchecked:
        run.send(f"ALERT: no fresh prices for {', '.join(unchecked)} - stops not checked")

    ledger = L.fill_pending(ledger, prices, as_of)
    ledger, events = L.process_exits(ledger, prices, as_of)
    for e in events:
        if e["kind"] != "closed":
            continue
        row = ledger[(ledger["ticker"] == e["ticker"]) & (ledger["pick_date"] == e["pick_date"])
                     & (ledger["status"] == "closed")]
        if len(row):
            e["pnl"] = float(row["pnl"].iloc[-1])
    res["events"] = events
    _save_ledger(run, ledger, paths["ledger"])

    equity = _valued_equity(ledger, prices, float(cfg["paper_account"]["start_equity"]))
    peak = max(_float(state.get("peak_equity")), equity)
    state["peak_equity"] = peak
    _save_json(run, paths["state"], state)

    run.step = "daily message"
    open_rows = ledger[ledger["status"] == "open"]
    account = {"equity": equity, "peak": peak,
               "drawdown_pct": (1 - equity / peak) * 100 if peak > 0 else 0.0}
    _send_main(run, state, paths, res,
               format_daily(events, open_rows, L.summary_by_version(ledger),
                            positions=_position_values(open_rows, prices), account=account))

    if prices:
        run.step = "archive"
        d = pd.Timestamp(day)
        target = root / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}" / "positions_prices.parquet"
        if not target.exists():  # a same-day re-run has already archived these prices
            rows = _write_archive(run, day,
                                  {"positions_prices": _prices_frame(prices, prices)}, root)
            res["archive_ok"] = _upload_archive(run, day, rows, root, paths["manifest"])

    run.step = "sheets mirror"
    _mirror(run, _mirror_tabs(ledger))
    return res
