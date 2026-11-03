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
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from scanner import ledger as L
from scanner.archive import append_manifest, write_snapshot
from scanner.config import ROOT, active_settings
from scanner.fetch import ANALYST_COLS, FUNDAMENTAL_COLS
from scanner.notify import FOOTER, apply_commands, format_daily, format_weekly
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

    common = dict(fetch_prices=fetch.fetch_prices, fetch_fundamentals=fetch.fetch_fundamentals,
                  fetch_analyst=fetch.fetch_analyst, fetch_earnings=fetch.fetch_earnings,
                  fetch_sp500=universe.fetch_sp500, secrets=secret_values(env))
    if dry_run:
        return Deps(telegram=ConsoleNotifier(), drive_upload=None, sheets_mirror=None,
                    chat_id="0", **common)
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
        state = _read_json(paths["state"], {"peak_equity": None, "last_weekly_pick_date": None})
    except Exception as exc:
        run.critical("state read", exc)
    return ledger, settings, state


def _commands(run: _Run, cfg: dict, paths: dict[str, Path], ledger: pd.DataFrame,
              settings: dict):
    """Apply queued Telegram commands. Persist ledger and settings first, then the
    offset, then reply - so a crash never loses a command it already acknowledged."""
    try:
        offset = _read_json(paths["offset"], {"offset": None}).get("offset")
    except Exception as exc:
        run.fail("telegram offset read", exc)
        offset = None
    try:
        updates = run.deps.telegram.get_updates(offset)
    except Exception as exc:
        run.fail("telegram updates", exc)
        updates = []
    settings, ledger, next_offset, replies = apply_commands(
        updates, settings, ledger, cfg, run.deps.chat_id)
    _save_ledger(run, ledger, paths["ledger"])
    _save_json(run, paths["settings"], settings)
    if next_offset is not None:  # no updates -> keep the stored offset
        try:
            _write_json(paths["offset"], {"offset": int(next_offset)})
        except Exception as exc:
            run.fail("telegram offset write", exc)
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
    """Upload written files to Drive, then record the manifest (failed rows on error)."""
    ok = rows is not None
    if rows is not None and run.deps.drive_upload is not None:
        try:
            run.deps.drive_upload(root, [r["file"] for r in rows])
        except Exception as exc:
            run.fail("drive upload", exc)
            ok = False
    try:
        append_manifest(rows or [], manifest, failed=not ok, run_date=day)
    except Exception as exc:
        run.fail("manifest", exc)
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
            "sent": run.sent, "skipped_reason": None}


# ------------------------------------------------------------ weekly
def run_weekly(cfg: dict, deps: Deps, as_of: pd.Timestamp, state_dir: Path) -> dict:
    run = _Run(deps)
    try:
        return _weekly(run, cfg, _day(as_of), Path(state_dir))
    except _Abort:
        raise RuntimeError("; ".join(run.errors)) from None


def _weekly(run: _Run, cfg: dict, as_of: pd.Timestamp, state_dir: Path) -> dict:
    res = _result(run)
    paths = _paths(state_dir)
    day = as_of.strftime("%Y-%m-%d")
    ledger, settings, state = _load(run, paths)
    settings, ledger = _commands(run, cfg, paths, ledger, settings)

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
        eligible, rejects = liquidity_filter(uni_prices, uni["min_dollar_volume"],
                                             uni["min_price"], uni.get("window", 20))
        for t in constituents["ticker"]:
            if t not in prices:
                rejects[t] = "no_data"
        frames["prices"] = _prices_frame(prices, eligible)

        equity = L.equity(ledger, start_equity, prices)
        peak = _float(state.get("peak_equity"))
        paused = buys_paused(equity, peak, sz["pause_drawdown_pct"])
        picks, sizing, stops, skipped, to_ledger = pd.DataFrame(), [], [], {}, []
        if not paused:
            fund = (run.deps.fetch_fundamentals(eligible) if eligible
                    else pd.DataFrame(columns=FUNDAMENTAL_COLS))
            analyst = (run.deps.fetch_analyst(eligible) if eligible
                       else pd.DataFrame(columns=ANALYST_COLS))
            earnings = (run.deps.fetch_earnings(eligible) if eligible
                        else pd.Series(dtype="datetime64[ns]"))
            # One sector source for every ticker (GICS from the constituent list), so
            # the per-sector cap never counts one sector under two different labels.
            fund = fund.reindex(eligible).copy()
            fund["sector"] = [gics.get(t) for t in fund.index]
            frames["fundamentals"] = fund.rename_axis("ticker")
            frames["analyst"] = analyst.reindex(eligible).rename_axis("ticker")
            frames["earnings"] = (earnings.reindex(eligible).rename("next_earnings")
                                  .rename_axis("ticker").to_frame())
            scored = None
            if eligible:
                table = build_metric_table({t: prices[t] for t in eligible}, fund, analyst,
                                           earnings, as_of)
                candidates, gate_rejects = apply_gates(
                    table, as_of, gates["stale_days"], gates["earnings_blackout_days"])
                rejects.update(gate_rejects)
                if len(candidates):
                    scored = score_table(candidates, active)
            if scored is not None:
                ranked = select_top(scored, n=len(scored), max_per_sector=len(scored))
                scores_tab = _join_reasons(ranked)
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
                                         equity)
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
        _mirror(run, _mirror_tabs(ledger))
        return res

    _save_ledger(run, ledger, paths["ledger"])
    state["last_weekly_pick_date"] = day
    _save_json(run, paths["state"], state)
    res["picks"] = picks

    if paused:
        frames = {k: frames[k] for k in ("constituents", "prices")}
    root = _archive_root(cfg)
    rows = _write_archive(run, day, frames, root)

    if paused:
        drop = (1 - equity / peak) * 100
        run.send(f"Buys paused: equity is {drop:.1f}% below peak\n{FOOTER}")
    else:
        run.send(format_weekly(picks, active, sizing, skipped, stops))

    res["archive_ok"] = _upload_archive(run, day, rows, root, paths["manifest"])
    _mirror(run, _mirror_tabs(ledger, scores_tab))
    return res


# ------------------------------------------------------------ daily
def run_daily(cfg: dict, deps: Deps, as_of: pd.Timestamp, state_dir: Path) -> dict:
    run = _Run(deps)
    try:
        return _daily(run, cfg, _day(as_of), Path(state_dir))
    except _Abort:
        raise RuntimeError("; ".join(run.errors)) from None


def _daily(run: _Run, cfg: dict, as_of: pd.Timestamp, state_dir: Path) -> dict:
    res = _result(run)
    paths = _paths(state_dir)
    day = as_of.strftime("%Y-%m-%d")
    ledger, settings, state = _load(run, paths)
    settings, ledger = _commands(run, cfg, paths, ledger, settings)

    tickers = list(dict.fromkeys(_active_rows(ledger)["ticker"]))
    prices: dict[str, pd.DataFrame] = {}
    fetched = True
    if tickers:
        try:
            prices = _fetch_prices(run.deps, tickers, as_of)
        except Exception as exc:
            run.fail("fetch prices", exc)
            fetched = False

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

    if fetched:
        mismatch = {e["ticker"] for e in events if e["kind"] == "data_mismatch"}
        valued = ledger[~((ledger["status"] == "open") & ledger["ticker"].isin(mismatch))]
        equity = L.equity(valued, float(cfg["paper_account"]["start_equity"]), prices)
        state["peak_equity"] = max(_float(state.get("peak_equity")), equity)
        _save_json(run, paths["state"], state)

    open_rows = ledger[ledger["status"] == "open"]
    run.send(format_daily(events, open_rows, L.summary_by_version(ledger)))

    if prices:
        root = _archive_root(cfg)
        d = pd.Timestamp(day)
        target = root / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}" / "positions_prices.parquet"
        if not target.exists():  # a same-day re-run has already archived these prices
            rows = _write_archive(run, day,
                                  {"positions_prices": _prices_frame(prices, prices)}, root)
            res["archive_ok"] = _upload_archive(run, day, rows, root, paths["manifest"])

    _mirror(run, _mirror_tabs(ledger))
    return res
