import copy
import json

import numpy as np
import pandas as pd
import pytest

from scanner.archive import MANIFEST_COLUMNS
from scanner.config import active_settings, load_scanner_config
from scanner.ledger import COLUMNS, read_ledger, write_ledger
from trader.data import synthetic_prices

BASE_CFG = load_scanner_config()
AS_OF = pd.Timestamp("2026-10-30")  # a Friday
TICKERS = [f"T{i:02d}" for i in range(12)]
SECTORS = ["Energy", "Health Care", "Industrials", "Utilities"]
SEED = 11  # with this seed, T00/T02/T05/T07/T11 pass the trend and momentum gates


# ----------------------------------------------------------------- fakes
class FakeTelegram:
    def __init__(self, updates=None, fail_send=False):
        self.updates = list(updates or [])
        self.fail_send = fail_send
        self.sent: list[str] = []
        self.offsets: list = []

    def send(self, text):
        if self.fail_send:
            raise RuntimeError("telegram down")
        self.sent.append(text)

    def get_updates(self, offset):
        self.offsets.append(offset)
        return [u for u in self.updates if offset is None or u["update_id"] >= offset]


def _fundamentals(tickers):
    rng = np.random.default_rng(3)
    n = len(tickers)
    return pd.DataFrame({"pe": rng.uniform(8, 40, n), "ev_ebitda": rng.uniform(5, 25, n),
                         "fcf_yield": rng.uniform(0.0, 0.08, n),
                         "sector": ["Technology"] * n}, index=list(tickers))


def _analyst(tickers):
    rng = np.random.default_rng(4)
    n = len(tickers)
    return pd.DataFrame({"rec_mean": rng.uniform(1.5, 3.5, n),
                         "target_gap": rng.uniform(-0.1, 0.3, n),
                         "rec_trend": rng.uniform(-0.1, 0.1, n)}, index=list(tickers))


def _earnings(tickers):
    return pd.Series(pd.NaT, index=list(tickers), dtype="datetime64[ns]")


def _sp500():
    return pd.DataFrame({"ticker": TICKERS,
                         "sector": [SECTORS[i % len(SECTORS)] for i in range(len(TICKERS))]})


class Recorder:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def __call__(self, *args):
        self.calls.append(args)
        if self.fail is not None:
            raise self.fail


_DEFAULT = object()


def make_deps(telegram=None, prices=None, future_bar=False, drive=_DEFAULT, sheets=None,
              price_calls=None, secrets=(), dry_run=False):
    from scanner.pipeline import Deps

    data = prices if prices is not None else synthetic_prices(
        TICKERS, start="2024-06-03", end="2026-10-30", seed=SEED)

    def fetch_prices(tickers, start):
        if price_calls is not None:
            price_calls.append((list(tickers), start))
        out = {}
        for t in tickers:
            if t not in data:
                continue
            df = data[t]
            if future_bar:
                extra = pd.DataFrame({"Open": [1e6], "High": [1e6], "Low": [1e6],
                                      "Close": [1e6], "Volume": [1e9]},
                                     index=[AS_OF + pd.Timedelta(days=3)])
                df = pd.concat([df, extra])
            out[t] = df
        return out

    return Deps(
        fetch_prices=fetch_prices,
        fetch_fundamentals=_fundamentals,
        fetch_analyst=_analyst,
        fetch_earnings=_earnings,
        fetch_sp500=_sp500,
        telegram=telegram if telegram is not None else FakeTelegram(),
        drive_upload=Recorder() if drive is _DEFAULT else drive,
        sheets_mirror=sheets if sheets is not None else Recorder(),
        chat_id="42",
        secrets=tuple(secrets),
        dry_run=dry_run,
    )


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(BASE_CFG)
    c["archive"]["archive_dir"] = str(tmp_path / "archive")  # absolute: ROOT / abs == abs
    return c


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


def weekly_messages(tg):
    return [m for m in tg.sent if m.startswith("Weekly picks") or m.startswith("No picks")]


def read_manifest(state_dir):
    return pd.read_csv(state_dir / "manifest.csv")


# ----------------------------------------------------------------- weekly
def test_weekly_end_to_end_offline(cfg, state_dir, tmp_path):
    from scanner.pipeline import run_weekly

    tg, drive, sheets = FakeTelegram(), Recorder(), Recorder()
    res = run_weekly(cfg, make_deps(tg, drive=drive, sheets=sheets), AS_OF, state_dir)

    assert res["errors"] == []
    assert res["archive_ok"] is True
    picks = res["picks"]
    assert 1 <= len(picks) <= 3
    ledger = read_ledger(state_dir / "ledger.csv")
    assert len(ledger) == len(picks)
    assert set(ledger["status"]) == {"pending"}
    assert set(ledger["ticker"]) == set(picks.index)
    assert set(ledger["pick_date"]) == {"2026-10-30"}
    assert len(weekly_messages(tg)) == 1
    assert tg.sent[-1].startswith("Weekly picks")
    assert res["sent"] == tg.sent

    day = tmp_path / "archive" / "2026" / "10" / "30"
    for name in ["constituents", "prices", "fundamentals", "analyst", "earnings", "scores",
                 "rejects"]:
        assert (day / f"{name}.parquet").exists(), name
    manifest = read_manifest(state_dir)
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == 7
    assert manifest["archive_ok"].astype(str).eq("True").all()

    # Drive got exactly the archived rel paths, under the archive root.
    (root, rels), = drive.calls
    assert root == tmp_path / "archive"
    assert sorted(rels) == sorted(manifest["file"])
    # Sheets mirror got the four weekly tabs; reasons are joined into a string.
    (tabs,), = sheets.calls
    assert set(tabs) == {"Picks", "Open", "WinLoss", "Scores"}
    assert len(tabs["Picks"]) == len(picks)
    assert tabs["Scores"]["reasons"].map(lambda v: isinstance(v, str)).all()
    assert "ticker" in tabs["Scores"].columns
    # Sector cap uses the GICS sector from the constituent list.
    assert set(picks["sector"]) <= set(SECTORS)

    state = json.loads((state_dir / "state.json").read_text())
    assert state["last_weekly_pick_date"] == "2026-10-30"


def test_weekly_rerun_same_week_no_duplicates_and_no_second_message(cfg, state_dir):
    from scanner.pipeline import run_weekly

    tg = FakeTelegram()
    first = run_weekly(cfg, make_deps(tg), AS_OF, state_dir)
    rows_before = read_ledger(state_dir / "ledger.csv")
    manifest_before = read_manifest(state_dir)

    second = run_weekly(cfg, make_deps(tg), AS_OF, state_dir)

    assert len(first["picks"]) >= 1
    assert second["picks"].empty
    assert second["errors"] == []
    assert "2026-10-30" in second["skipped_reason"]
    assert len(weekly_messages(tg)) == 1
    pd.testing.assert_frame_equal(read_ledger(state_dir / "ledger.csv"), rows_before)
    assert len(read_manifest(state_dir)) == len(manifest_before)


def test_weekly_sends_no_picks_message_when_nothing_passes(cfg, state_dir):
    from scanner.pipeline import run_weekly

    cfg["universe"]["min_dollar_volume"] = 1e15
    tg = FakeTelegram()
    res = run_weekly(cfg, make_deps(tg), AS_OF, state_dir)

    assert res["picks"].empty
    assert res["errors"] == []
    assert len(weekly_messages(tg)) == 1
    assert tg.sent[-1].startswith("No picks this week")
    assert read_ledger(state_dir / "ledger.csv").empty

    # A no-picks week writes no ledger rows, but a re-run still must not resend.
    again = run_weekly(cfg, make_deps(tg), AS_OF, state_dir)
    assert again["skipped_reason"]
    assert len(weekly_messages(tg)) == 1


def test_archive_failure_alerts_but_telegram_and_ledger_still_done(cfg, state_dir):
    from scanner.pipeline import run_weekly

    tg = FakeTelegram()
    drive = Recorder(fail=RuntimeError("drive down"))
    res = run_weekly(cfg, make_deps(tg, drive=drive), AS_OF, state_dir)

    assert res["archive_ok"] is False
    assert any("drive down" in e for e in res["errors"])
    assert len(read_ledger(state_dir / "ledger.csv")) == len(res["picks"]) > 0
    assert len(weekly_messages(tg)) == 1
    assert any(m.startswith("ALERT: drive upload failed") for m in tg.sent)
    manifest = read_manifest(state_dir)
    assert manifest["archive_ok"].astype(str).eq("False").all()


def test_archive_write_failure_records_failed_manifest_row(cfg, state_dir, tmp_path):
    from scanner.pipeline import run_weekly

    # Pre-create one target so write_snapshot refuses (append-only archive).
    day = tmp_path / "archive" / "2026" / "10" / "30"
    day.mkdir(parents=True)
    (day / "prices.parquet").write_bytes(b"x")
    tg = FakeTelegram()
    res = run_weekly(cfg, make_deps(tg), AS_OF, state_dir)

    assert res["archive_ok"] is False
    assert any(m.startswith("ALERT: archive failed") for m in tg.sent)
    manifest = read_manifest(state_dir)
    assert len(manifest) == 1
    assert str(manifest["archive_ok"].iloc[0]) == "False"
    assert manifest["run_date"].iloc[0] == "2026-10-30"
    assert len(weekly_messages(tg)) == 1


def test_sheets_failure_does_not_block(cfg, state_dir):
    from scanner.pipeline import run_weekly

    tg = FakeTelegram()
    sheets = Recorder(fail=RuntimeError("quota exceeded"))
    res = run_weekly(cfg, make_deps(tg, sheets=sheets), AS_OF, state_dir)

    assert any("quota exceeded" in e for e in res["errors"])
    assert res["archive_ok"] is True
    assert len(read_ledger(state_dir / "ledger.csv")) > 0
    assert len(weekly_messages(tg)) == 1
    assert any(m.startswith("ALERT: sheets mirror failed") for m in tg.sent)


def test_secret_values_are_scrubbed_from_errors_and_alerts(cfg, state_dir):
    from scanner.pipeline import run_weekly

    secret = "SHEET-ID-123456"
    tg = FakeTelegram()
    sheets = Recorder(fail=RuntimeError(f"<HttpError 400 for spreadsheets/{secret}/values>"))
    res = run_weekly(cfg, make_deps(tg, sheets=sheets, secrets=[secret]), AS_OF, state_dir)

    assert res["errors"]
    assert not any(secret in e for e in res["errors"])
    assert not any(secret in m for m in tg.sent)
    assert any(m.startswith("ALERT: sheets mirror failed") for m in tg.sent)


def test_telegram_send_failure_is_recorded_without_loop(cfg, state_dir):
    from scanner.pipeline import run_weekly

    tg = FakeTelegram(fail_send=True)
    res = run_weekly(cfg, make_deps(tg), AS_OF, state_dir)

    assert any("telegram down" in e for e in res["errors"])
    assert res["sent"] == []
    assert len(read_ledger(state_dir / "ledger.csv")) > 0
    assert res["archive_ok"] is True


def test_buys_paused_when_drawdown_exceeds_limit_still_sends_alerts(cfg, state_dir):
    from scanner.pipeline import run_daily, run_weekly

    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(json.dumps(
        {"peak_equity": 2000.0, "last_weekly_pick_date": None}))
    tg, sheets = FakeTelegram(), Recorder()
    res = run_weekly(cfg, make_deps(tg, sheets=sheets), AS_OF, state_dir)

    assert res["picks"].empty
    assert res["errors"] == []
    assert [m for m in tg.sent if m.startswith("Buys paused")] == [
        tg.sent[-1]]
    assert "50.0% below peak" in tg.sent[-1]
    assert weekly_messages(tg) == []
    assert read_ledger(state_dir / "ledger.csv").empty
    assert len(sheets.calls) == 1  # mirror still runs
    # Weekly only reads the peak.
    assert json.loads((state_dir / "state.json").read_text())["peak_equity"] == 2000.0

    daily = run_daily(cfg, make_deps(tg), AS_OF, state_dir)
    assert daily["errors"] == []
    assert tg.sent[-1].startswith("Daily update")


def test_commands_applied_before_scoring(cfg, state_dir):
    from scanner.pipeline import run_daily, run_weekly

    updates = [{"update_id": 5, "message": {"text": "/profile momentum", "chat": {"id": 42}}}]
    tg = FakeTelegram(updates)
    res = run_weekly(cfg, make_deps(tg), AS_OF, state_dir)

    momentum = active_settings(cfg, {"profile": "momentum"})["version"]
    ledger = read_ledger(state_dir / "ledger.csv")
    assert len(ledger) > 0
    assert set(ledger["config_version"]) == {momentum}
    assert json.loads((state_dir / "settings.json").read_text())["profile"] == "momentum"
    assert json.loads((state_dir / "telegram_offset.json").read_text())["offset"] == 6
    assert tg.sent[0] == "profile set to Momentum"
    assert "Momentum" in weekly_messages(tg)[0]
    assert len(res["picks"]) >= 1

    # A later run with no new updates keeps the stored offset.
    run_daily(cfg, make_deps(tg), AS_OF, state_dir)
    assert tg.offsets[-1] == 6
    assert json.loads((state_dir / "telegram_offset.json").read_text())["offset"] == 6


def test_future_bars_are_not_used(cfg, tmp_path):
    from scanner.pipeline import run_weekly

    control = run_weekly(cfg, make_deps(FakeTelegram()), AS_OF, tmp_path / "a")
    cfg2 = copy.deepcopy(cfg)
    cfg2["archive"]["archive_dir"] = str(tmp_path / "archive2")
    leaked = run_weekly(cfg2, make_deps(FakeTelegram(), future_bar=True), AS_OF, tmp_path / "b")

    assert len(control["picks"]) >= 1
    assert list(leaked["picks"].index) == list(control["picks"].index)
    np.testing.assert_allclose(leaked["picks"]["close"], control["picks"]["close"])
    assert (leaked["picks"]["close"] < 1e5).all()


# ----------------------------------------------------------------- daily
def _bars(end, n, price=100.0):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price,
                         "Volume": 1_000_000.0}, index=idx)


def _set(df, day, **vals):
    for k, v in vals.items():
        df.loc[pd.Timestamp(day), k] = v
    return df


def _ledger_row(**kw):
    row = {c: "" for c in COLUMNS}
    row.update({"config_version": "abc12345", "took_it": False, "entry_price": np.nan,
                "stop": np.nan, "quantity": np.nan, "exit_price": np.nan, "pnl": np.nan,
                "score_snapshot": "{}", "data_timestamp": ""})
    row.update(kw)
    return row


def _write_rows(state_dir, rows):
    write_ledger(pd.DataFrame(rows, columns=COLUMNS), state_dir / "ledger.csv")


DAILY_AS_OF = pd.Timestamp("2026-10-09")


def test_daily_fills_pending_then_exits_on_stop(cfg, state_dir):
    from scanner.pipeline import run_daily

    df = _bars(DAILY_AS_OF, 80)
    _set(df, "2026-10-02", Low=99.0)
    _set(df, "2026-10-05", Open=98.0, Low=85.0, Close=88.0, High=98.0)
    for d in ["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]:
        _set(df, d, Open=88.0, High=88.0, Low=88.0, Close=88.0)
    _write_rows(state_dir, [_ledger_row(pick_date="2026-10-01", ticker="AAA",
                                        status="pending", stop=90.0, quantity=1.0)])
    tg, sheets = FakeTelegram(), Recorder()
    res = run_daily(cfg, make_deps(tg, prices={"AAA": df}, sheets=sheets), DAILY_AS_OF,
                    state_dir)

    ledger = read_ledger(state_dir / "ledger.csv")
    row = ledger.iloc[0]
    assert row["status"] == "closed"
    assert row["entry_date"] == "2026-10-02"
    assert row["entry_price"] == 100.0
    assert row["exit_reason"] == "stop"
    assert row["exit_price"] == 90.0
    assert [e["kind"] for e in res["events"]] == ["stop"]
    assert res["errors"] == []
    assert tg.sent[-1].startswith("Daily update")
    assert "Stop hit: AAA at 90.00" in tg.sent[-1]
    (tabs,), = sheets.calls
    assert set(tabs) == {"Picks", "Open", "WinLoss"}
    assert res["archive_ok"] is True
    assert (state_dir / "manifest.csv").exists()
    manifest = read_manifest(state_dir)
    assert manifest["file"].iloc[0].endswith("positions_prices.parquet")


def test_daily_rerun_same_day_does_not_alert_on_existing_archive(cfg, state_dir):
    from scanner.pipeline import run_daily

    df = _bars(DAILY_AS_OF, 80)
    _write_rows(state_dir, [_ledger_row(pick_date="2026-10-01", ticker="AAA",
                                        status="pending", stop=90.0, quantity=1.0)])
    tg = FakeTelegram()
    run_daily(cfg, make_deps(tg, prices={"AAA": df}), DAILY_AS_OF, state_dir)
    again = run_daily(cfg, make_deps(tg, prices={"AAA": df}), DAILY_AS_OF, state_dir)
    assert again["errors"] == []
    assert again["archive_ok"] is True
    assert not any(m.startswith("ALERT") for m in tg.sent)


def test_closed_event_message_shows_pnl(cfg, state_dir):
    from scanner.pipeline import run_daily

    df = _bars(DAILY_AS_OF, 80)
    _set(df, "2026-10-05", Open=99.0, High=99.0, Low=94.0, Close=95.0)
    for d in ["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]:
        _set(df, d, Open=96.0, High=96.0, Low=96.0, Close=96.0)
    _write_rows(state_dir, [_ledger_row(pick_date="2026-10-01", ticker="AAA", status="open",
                                        entry_date="2026-10-02", entry_price=100.0,
                                        stop=50.0, quantity=2.0)])
    tg = FakeTelegram()
    res = run_daily(cfg, make_deps(tg, prices={"AAA": df}), DAILY_AS_OF, state_dir)

    closed = [e for e in res["events"] if e["kind"] == "closed"]
    assert len(closed) == 1 and closed[0]["pnl"] == pytest.approx(-8.0)
    assert "Closed: AAA at 96.00 (sell_signal), P&L -8.00" in tg.sent[-1]


def test_daily_with_no_open_positions_sends_no_changes(cfg, state_dir):
    from scanner.pipeline import run_daily

    calls = []
    tg = FakeTelegram()
    res = run_daily(cfg, make_deps(tg, price_calls=calls), DAILY_AS_OF, state_dir)

    assert calls == []
    assert res["errors"] == []
    assert res["events"] == []
    assert "No changes today." in tg.sent[-1]


def test_data_mismatch_row_excluded_from_equity(cfg, state_dir):
    from scanner.pipeline import run_daily

    df = _bars(DAILY_AS_OF, 80, price=300.0)
    _set(df, "2026-10-02", Open=150.0)
    _write_rows(state_dir, [_ledger_row(pick_date="2026-10-01", ticker="BBB", status="open",
                                        entry_date="2026-10-02", entry_price=100.0,
                                        stop=50.0, quantity=1.0)])
    tg = FakeTelegram()
    res = run_daily(cfg, make_deps(tg, prices={"BBB": df}), DAILY_AS_OF, state_dir)

    assert [e["kind"] for e in res["events"]] == ["data_mismatch"]
    state = json.loads((state_dir / "state.json").read_text())
    # Including the mis-valued row would give 1000 + 1 * (300 - 100) = 1200.
    assert state["peak_equity"] == pytest.approx(cfg["paper_account"]["start_equity"])
    assert "ALERT: BBB price data looks inconsistent" in tg.sent[-1]


def test_daily_slices_future_bars(cfg, state_dir):
    from scanner.pipeline import run_daily

    df = _bars(DAILY_AS_OF, 80)
    df.loc[DAILY_AS_OF + pd.Timedelta(days=3)] = [100.0, 1e6, 1.0, 1e6, 1e6]
    _write_rows(state_dir, [_ledger_row(pick_date="2026-10-01", ticker="AAA", status="open",
                                        entry_date="2026-10-02", entry_price=100.0,
                                        stop=90.0, quantity=1.0)])
    res = run_daily(cfg, make_deps(FakeTelegram(), prices={"AAA": df}), DAILY_AS_OF, state_dir)
    assert res["events"] == []
    assert read_ledger(state_dir / "ledger.csv").iloc[0]["status"] == "open"
    # Equity has no cutoff of its own: a future Close would inflate the peak.
    state = json.loads((state_dir / "state.json").read_text())
    assert state["peak_equity"] == pytest.approx(cfg["paper_account"]["start_equity"])


# ----------------------------------------------------------------- deps / CLI
def test_missing_telegram_env_raises_systemexit_without_values():
    from scanner.pipeline import default_deps

    env = {"TELEGRAM_BOT_TOKEN": "123:TOPSECRET", "SHEET_ID": "SHEETSECRET"}
    with pytest.raises(SystemExit) as exc:
        default_deps(env, dry_run=False)
    msg = str(exc.value)
    assert "TELEGRAM_CHAT_ID" in msg
    assert "TOPSECRET" not in msg and "SHEETSECRET" not in msg


def test_dry_run_uses_console_and_no_google(capsys):
    from scanner.pipeline import default_deps

    env = {"GOOGLE_SERVICE_ACCOUNT_JSON": "{}", "DRIVE_FOLDER_ID": "f", "SHEET_ID": "s"}
    deps = default_deps(env, dry_run=True)
    assert deps.drive_upload is None and deps.sheets_mirror is None
    assert deps.chat_id == "0"
    assert deps.telegram.get_updates(None) == []
    deps.telegram.send("hello there")
    assert "hello there" in capsys.readouterr().out


def test_real_deps_without_google_env_skip_google():
    from scanner.notify import TelegramClient
    from scanner.pipeline import default_deps

    env = {"TELEGRAM_BOT_TOKEN": "123:TOPSECRET", "TELEGRAM_CHAT_ID": "42"}
    deps = default_deps(env, dry_run=False)
    assert isinstance(deps.telegram, TelegramClient)
    assert deps.chat_id == "42"
    assert deps.drive_upload is None and deps.sheets_mirror is None
    assert "123:TOPSECRET" in deps.secrets


def test_cli_parses_arguments():
    from scanner.cli import parse_args, resolve_as_of

    a = parse_args(["weekly", "--as-of", "2026-10-30", "--dry-run", "--state-dir", "x"])
    assert a.command == "weekly" and a.dry_run is True and a.state_dir == "x"
    assert resolve_as_of(a.as_of) == pd.Timestamp("2026-10-30")
    b = parse_args(["daily"])
    assert b.command == "daily" and b.dry_run is False and b.as_of is None
    assert b.state_dir is None
    today = resolve_as_of(None)
    assert today == today.normalize() and today.tzinfo is None
    assert parse_args(["check-drive"]).command == "check-drive"
    with pytest.raises(SystemExit):
        parse_args(["monthly"])
    with pytest.raises(SystemExit):
        parse_args(["weekly", "--as-of", "30/10/2026"])


def test_cli_check_drive_reports_error_and_exit_code(monkeypatch, capsys):
    from scanner import cli

    monkeypatch.setattr(cli, "_check_drive", lambda env: (_ for _ in ()).throw(
        RuntimeError("Drive write failed: not shared")))
    assert cli.main(["check-drive"], env={}) == 1
    assert "not shared" in capsys.readouterr().out
    monkeypatch.setattr(cli, "_check_drive", lambda env: "file-id-1")
    assert cli.main(["check-drive"], env={}) == 0
    assert "file-id-1" in capsys.readouterr().out


# ----------------------------------------------------------------- review fixes (round 1)
def _synthetic():
    return synthetic_prices(TICKERS, start="2024-06-03", end="2026-10-30", seed=SEED)


def _open_row(df, ticker, entry_date, entry_price=None, qty=0.1, stop=None):
    entry = float(df.loc[pd.Timestamp(entry_date), "Open"]) if entry_price is None else entry_price
    return _ledger_row(pick_date="2026-08-31", ticker=ticker, status="open",
                       entry_date=entry_date, entry_price=entry,
                       stop=entry * 0.9 if stop is None else stop, quantity=qty)


def test_dry_run_default_paths_never_touch_real_state(tmp_path, monkeypatch):
    from scanner import cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    tg = FakeTelegram()
    deps = make_deps(tg, drive=None, dry_run=True)
    deps.sheets_mirror = None
    monkeypatch.setattr(cli, "default_deps", lambda env, dry_run: deps)

    assert cli.main(["weekly", "--dry-run", "--as-of", "2026-10-30"], env={}) == 0

    assert not (tmp_path / "scanner_state").exists()
    assert not (tmp_path / "data").exists()
    assert (tmp_path / ".dryrun" / "state" / "ledger.csv").exists()
    assert (tmp_path / ".dryrun" / "archive" / "2026" / "10" / "30" / "prices.parquet").exists()
    # A dry run with no Drive is not an error.
    assert not any(m.startswith("ALERT") for m in tg.sent)

    args = cli.parse_args(["daily", "--dry-run", "--state-dir", str(tmp_path / "mine")])
    state_dir, archive_root = cli.resolve_paths(args, BASE_CFG)
    assert state_dir == tmp_path / "mine"
    assert archive_root == tmp_path / ".dryrun" / "archive"
    state_dir, archive_root = cli.resolve_paths(cli.parse_args(["daily"]), BASE_CFG)
    assert state_dir == tmp_path / BASE_CFG["archive"]["state_dir"]
    assert archive_root == tmp_path / BASE_CFG["archive"]["archive_dir"]


def test_held_ticker_is_not_repicked(cfg, tmp_path):
    from scanner.pipeline import run_weekly

    control = run_weekly(cfg, make_deps(FakeTelegram()), AS_OF, tmp_path / "a")
    first = control["picks"].index[0]

    data = _synthetic()
    state_b = tmp_path / "b"
    _write_rows(state_b, [_open_row(data[first], first, "2026-09-01")])
    cfg2 = copy.deepcopy(cfg)
    cfg2["archive"]["archive_dir"] = str(tmp_path / "archive2")
    res = run_weekly(cfg2, make_deps(FakeTelegram()), AS_OF, state_b)

    assert first not in res["picks"].index
    assert len(res["picks"]) >= 1
    rejects = pd.read_parquet(tmp_path / "archive2" / "2026" / "10" / "30" / "rejects.parquet")
    assert rejects.set_index("ticker").loc[first, "reason"] == "already held"
    ledger = read_ledger(state_b / "ledger.csv")
    assert (ledger["ticker"] == first).sum() == 1


def test_split_adjusted_held_position_does_not_pause_or_shrink(cfg, tmp_path):
    from scanner.pipeline import run_weekly

    def seed_state(d):
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"peak_equity": 1000.0,
                                                  "last_weekly_pick_date": None}))

    seed_state(tmp_path / "a")
    control = run_weekly(cfg, make_deps(FakeTelegram()), AS_OF, tmp_path / "a")

    # T01 fails the trend gate anyway, so it cannot change the picks by itself.
    data = _synthetic()
    pre_split = 2 * float(data["T01"].loc[pd.Timestamp("2026-09-01"), "Open"])
    state_b = tmp_path / "b"
    seed_state(state_b)
    _write_rows(state_b, [_open_row(data["T01"], "T01", "2026-09-01", entry_price=pre_split,
                                    qty=3.0, stop=pre_split - 1)])
    cfg2 = copy.deepcopy(cfg)
    cfg2["archive"]["archive_dir"] = str(tmp_path / "archive2")
    tg = FakeTelegram()
    res = run_weekly(cfg2, make_deps(tg), AS_OF, state_b)

    assert not any(m.startswith("Buys paused") for m in tg.sent)
    assert list(res["picks"].index) == list(control["picks"].index)
    qa = read_ledger(tmp_path / "a" / "ledger.csv").set_index("ticker")["quantity"]
    qb = read_ledger(state_b / "ledger.csv").set_index("ticker")["quantity"]
    for t in control["picks"].index:
        assert qb[t] == pytest.approx(qa[t])


def test_corrupt_offset_file_aborts_without_get_updates(cfg, state_dir):
    from scanner.pipeline import run_weekly

    state_dir.mkdir(parents=True)
    (state_dir / "telegram_offset.json").write_text("{not json")
    tg = FakeTelegram()
    with pytest.raises(RuntimeError):
        run_weekly(cfg, make_deps(tg), AS_OF, state_dir)
    assert tg.offsets == []
    assert any(m.startswith("ALERT: telegram offset read failed") for m in tg.sent)
    assert not (state_dir / "ledger.csv").exists()


def test_offset_persisted_before_ledger_save_failure(cfg, state_dir, monkeypatch):
    from scanner import ledger as ledger_mod
    from scanner.pipeline import run_daily

    def boom(df, path):
        raise OSError("disk full")

    monkeypatch.setattr(ledger_mod, "write_ledger", boom)
    updates = [{"update_id": 9, "message": {"text": "/toggle rsi", "chat": {"id": 42}}}]
    tg = FakeTelegram(updates)
    with pytest.raises(RuntimeError):
        run_daily(cfg, make_deps(tg), DAILY_AS_OF, state_dir)
    assert json.loads((state_dir / "telegram_offset.json").read_text())["offset"] == 10
    assert any(m.startswith("ALERT: ledger write failed") for m in tg.sent)
    assert not any(m.startswith("rsi") for m in tg.sent)  # no reply before state is saved


class FixedTelegram(FakeTelegram):
    """Returns its updates whatever the offset (e.g. update_id restarted at random)."""

    def get_updates(self, offset):
        self.offsets.append(offset)
        return list(self.updates)


def test_toggle_delivered_twice_flips_once(cfg, state_dir, monkeypatch):
    """Crash after the toggle was applied in memory: at most once means the toggle is
    lost (settings never saved) but the offset has moved, so it is never re-applied."""
    from scanner import ledger as ledger_mod
    from scanner.pipeline import run_daily

    updates = [{"update_id": 7, "message": {"text": "/toggle rsi", "chat": {"id": 42}}}]
    tg = FakeTelegram(updates)  # honours the offset, like Telegram

    def boom(df, path):
        raise OSError("disk full")

    with monkeypatch.context() as mp:
        mp.setattr(ledger_mod, "write_ledger", boom)
        with pytest.raises(RuntimeError):
            run_daily(cfg, make_deps(tg), DAILY_AS_OF, state_dir)
    assert json.loads((state_dir / "telegram_offset.json").read_text())["offset"] == 8

    run_daily(cfg, make_deps(tg), DAILY_AS_OF, state_dir)

    assert tg.offsets == [None, 8]
    settings = json.loads((state_dir / "settings.json").read_text())
    assert "toggles" not in settings  # preset unchanged: toggle lost, never flipped twice
    assert not any(m.startswith("rsi") for m in tg.sent)


def test_small_update_id_after_offset_reset_is_still_processed(cfg, state_dir):
    from scanner.pipeline import run_daily

    state_dir.mkdir(parents=True)
    (state_dir / "telegram_offset.json").write_text(json.dumps({"offset": 5000}))
    updates = [{"update_id": 17, "message": {"text": "/toggle rsi", "chat": {"id": 42}}}]
    tg = FixedTelegram(updates)
    run_daily(cfg, make_deps(tg), DAILY_AS_OF, state_dir)

    assert tg.offsets == [5000]
    assert "rsi" not in json.loads((state_dir / "settings.json").read_text())["toggles"]
    assert json.loads((state_dir / "telegram_offset.json").read_text())["offset"] == 18


def test_malformed_update_ids_are_skipped_not_fatal(cfg, state_dir):
    from scanner.pipeline import run_daily

    updates = [{"message": {"text": "/profile value", "chat": {"id": 42}}},
               {"update_id": "x", "message": {"text": "/profile value", "chat": {"id": 42}}},
               {"update_id": 3, "message": {"text": "/toggle rsi", "chat": {"id": 42}}}]
    tg = FixedTelegram(updates)
    res = run_daily(cfg, make_deps(tg), DAILY_AS_OF, state_dir)

    assert res["errors"] == []
    settings = json.loads((state_dir / "settings.json").read_text())
    assert settings.get("profile", "balanced") == "balanced"
    assert "rsi" not in settings["toggles"]
    assert json.loads((state_dir / "telegram_offset.json").read_text())["offset"] == 4


def test_unexpected_exception_alerts_and_propagates(cfg, state_dir, monkeypatch):
    from scanner import pipeline

    def broken(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(pipeline, "format_weekly", broken)
    tg = FakeTelegram()
    with pytest.raises(ValueError, match="boom"):
        pipeline.run_weekly(cfg, make_deps(tg), AS_OF, state_dir)
    assert "ALERT: weekly message failed - ValueError: boom" in tg.sent


def test_drive_not_configured_outside_dry_run_alerts(cfg, state_dir):
    from scanner.pipeline import run_weekly

    tg = FakeTelegram()
    res = run_weekly(cfg, make_deps(tg, drive=None), AS_OF, state_dir)

    assert res["archive_ok"] is False
    assert "Drive not configured; archive not uploaded" in res["errors"]
    alerts = [m for m in tg.sent if m.startswith("ALERT")]
    assert alerts == ["ALERT: drive upload failed - Drive not configured; archive not uploaded"]
    manifest = read_manifest(state_dir)
    assert len(manifest) == 7
    assert manifest["archive_ok"].astype(str).eq("True").all()  # local files are fine
    assert len(weekly_messages(tg)) == 1
