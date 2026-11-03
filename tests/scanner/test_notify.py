import pandas as pd
import pytest

from scanner.config import active_settings, load_scanner_config
from scanner.ledger import COLUMNS
from scanner.notify import (
    TelegramClient,
    apply_commands,
    format_daily,
    format_weekly,
    parse_command,
    split_message,
)

CFG = load_scanner_config()
TOKEN = "123456:SECRET-TOKEN-XYZ"


def picks_frame(earnings_known=True):
    df = pd.DataFrame(
        {
            "close": [100.0, 50.0], "atr14": [2.0, 1.0], "blend": [81.5, 70.25],
            "technical": [80.0, 70.0], "fundamental": [75.0, 60.0],
            "sentiment": [60.0, 55.0], "risk": [90.0, 85.0],
            "sector": ["Tech", "Energy"],
            "earnings_known": [earnings_known, True],
            "next_earnings": [None, None],
        },
        index=["AAA", "BBB"],
    )
    df["reasons"] = pd.Series([["strong trend", "cheap"], ["momentum"]], index=df.index, dtype=object)
    return df


SIZING = [
    {"quantity": 10, "value": 1000.0, "risk_amount": 40.0, "status": "ok"},
    {"quantity": 20, "value": 1000.0, "risk_amount": 40.0, "status": "ok"},
]


def test_format_weekly_lists_each_pick_with_scores_stop_quantity_and_version():
    settings = active_settings(CFG, {"profile": "balanced"})
    text = format_weekly(picks_frame(), settings, SIZING, {}, stops=[96.0, 48.0])
    assert settings["label"] in text and settings["version"] in text
    for t in ("AAA", "BBB", "Tech", "Energy", "81.5", "96.00", "48.00", "strong trend"):
        assert t in text
    assert "PAPER ONLY - not financial advice" in text
    assert "1." in text and "2." in text


def test_format_weekly_no_picks_message():
    settings = active_settings(CFG, {"profile": "balanced"})
    text = format_weekly(picks_frame().iloc[0:0], settings, [], {"ZZZ": "below minimum"})
    assert text.startswith("No picks this week")
    assert "ZZZ" in text and "below minimum" in text


def test_format_weekly_shows_earnings_unknown():
    settings = active_settings(CFG, {"profile": "balanced"})
    text = format_weekly(picks_frame(earnings_known=False), settings, SIZING, {}, stops=[96.0, 48.0])
    assert text.count("earnings date unknown") == 1
    ok = format_weekly(picks_frame(), settings, SIZING, {}, stops=[96.0, 48.0])
    assert "earnings date unknown" not in ok


def test_format_daily_data_mismatch_is_an_alert():
    ev = [{"ticker": "AAA", "pick_date": "2026-10-01", "kind": "data_mismatch",
           "price": None, "date": "2026-11-03", "detail": "gap"}]
    text = format_daily(ev, pd.DataFrame(), pd.DataFrame())
    assert "ALERT" in text and "AAA" in text and "check" in text.lower()


def test_format_daily_no_changes():
    summary = pd.DataFrame([{"config_version": "v1", "trades": 2, "wins": 1, "losses": 1,
                             "win_rate": 0.5, "total_pnl": 12.5}])
    text = format_daily([], pd.DataFrame(), summary)
    assert "No changes today." in text and "v1" in text and "50%" in text and "12.50" in text


def test_format_daily_events_and_open_positions():
    ev = [
        {"ticker": "TA", "pick_date": "d", "kind": "stop", "price": 9.0, "date": "x"},
        {"ticker": "TB", "pick_date": "d", "kind": "sell_signal", "price": 9.0, "date": "x"},
        {"ticker": "TC", "pick_date": "d", "kind": "time_exit", "price": 9.0, "date": "x"},
        {"ticker": "TD", "pick_date": "d", "kind": "closed", "price": 9.0, "date": "x",
         "exit_reason": "stop", "pnl": -5.0},
    ]
    op = pd.DataFrame([{"ticker": "TE", "quantity": 5, "entry_price": 10.0, "stop": 8.0, "took_it": True}])
    text = format_daily(ev, op, pd.DataFrame())
    for t in ("TA", "TB", "TC", "TD", "TE", "Stop hit", "Sell signal", "Time exit"):
        assert t in text
    assert "No changes today." not in text


def test_split_message_respects_limit_on_line_boundaries():
    text = "\n".join(["x" * 99] * 50)
    assert len(text) > 4096
    parts = split_message(text)
    assert all(len(p) <= 4096 for p in parts) and len(parts) > 1
    assert "\n".join(parts) == text
    long = "y" * 5000
    parts = split_message(long)
    assert all(len(p) <= 4096 for p in parts) and "".join(parts) == long


def test_parse_command_variants():
    assert parse_command("/profile Momentum") == {"cmd": "profile", "arg": "momentum"}
    assert parse_command("/toggle RSI") == {"cmd": "toggle", "arg": "rsi"}
    assert parse_command("/took aapl") == {"cmd": "took", "arg": "AAPL"}
    assert parse_command("/took@mybot aapl") == {"cmd": "took", "arg": "AAPL"}
    assert parse_command("/took") is None
    assert parse_command("hello") is None
    assert parse_command("/unknown x") is None


class FakeResp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, data=None, raises=None):
        self.calls = []
        self.data = {"ok": True, "result": []} if data is None else data
        self.raises = raises

    def _do(self, kind, url, **kw):
        self.calls.append((kind, url, kw))
        if self.raises:
            raise self.raises
        return FakeResp(self.data)

    def post(self, url, **kw):
        return self._do("post", url, **kw)

    def get(self, url, **kw):
        return self._do("get", url, **kw)


def test_telegram_send_posts_each_part():
    s = FakeSession()
    TelegramClient(TOKEN, "42", session=s).send("\n".join(["x" * 99] * 50))
    assert len(s.calls) == 2
    kind, url, kw = s.calls[0]
    assert kind == "post" and url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert kw["data"]["chat_id"] == "42" and "parse_mode" not in kw["data"]


def test_get_updates_returns_result():
    s = FakeSession({"ok": True, "result": [{"update_id": 1}]})
    out = TelegramClient(TOKEN, "42", session=s).get_updates(5)
    assert out == [{"update_id": 1}]
    assert s.calls[0][2]["params"] == {"offset": 5, "timeout": 0}


def test_token_never_in_error_message():
    c = TelegramClient(TOKEN, "42", session=FakeSession({"ok": False, "description": "bad"}))
    with pytest.raises(RuntimeError) as e:
        c.send("hi")
    assert TOKEN not in str(e.value)
    c = TelegramClient(TOKEN, "42", session=FakeSession(raises=ValueError(f"boom {TOKEN}")))
    with pytest.raises(RuntimeError) as e:
        c.send("hi")
    assert TOKEN not in str(e.value) and e.value.__cause__ is None
    with pytest.raises(RuntimeError) as e:
        c.get_updates(None)
    assert TOKEN not in str(e.value)
    assert TOKEN not in repr(c)


def upd(i, text, chat="42"):
    return {"update_id": i, "message": {"text": text, "chat": {"id": int(chat)}}}


def make_ledger():
    rows = [
        {"pick_date": "2026-09-01", "ticker": "AAPL", "took_it": False},
        {"pick_date": "2026-10-01", "ticker": "AAPL", "took_it": False},
    ]
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df:
            df[c] = None
    return df[COLUMNS]


def test_apply_commands_switches_profile_and_marks_custom_on_toggle():
    settings = {"profile": "balanced"}
    ledger = make_ledger()
    ups = [upd(2, "/toggle rsi"), upd(1, "/profile momentum"), upd(3, "/took aapl")]
    new, led, off, replies = apply_commands(ups, settings, ledger, CFG, "42")
    assert settings == {"profile": "balanced"}  # not mutated
    assert off == 4 and len(replies) == 3
    assert new["profile"] == "momentum" and "toggles" in new
    assert "rsi" not in new["toggles"]
    assert "Custom" in replies[1]
    assert led.loc[led["pick_date"] == "2026-10-01", "took_it"].all()
    assert not led.loc[led["pick_date"] == "2026-09-01", "took_it"].any()
    assert not ledger["took_it"].any()


def test_apply_commands_invalid_inputs():
    _, _, _, replies = apply_commands(
        [upd(1, "/profile nope"), upd(2, "/toggle nope"), upd(3, "/took ZZZ")],
        {"profile": "balanced"}, make_ledger(), CFG, "42")
    assert "value" in replies[0].lower() and "momentum" in replies[0].lower()
    assert "rsi" in replies[1].lower()
    assert replies[2] == "no pick found for ZZZ"


def test_apply_commands_ignores_other_chat_ids():
    new, _, _, replies = apply_commands(
        [upd(7, "/profile momentum", chat="99")], {"profile": "balanced"},
        make_ledger(), CFG, "42")
    assert new == {"profile": "balanced"} and replies == []


def test_next_offset_advances_past_foreign_chat_updates():
    ups = [upd(7, "/profile momentum", chat="99"), {"update_id": 9, "edited": {}}]
    _, _, off, _ = apply_commands(ups, {"profile": "balanced"}, make_ledger(), CFG, "42")
    assert off == 10
    assert apply_commands([], {"profile": "balanced"}, make_ledger(), CFG, "42")[2] is None


def test_toggle_back_to_preset_drops_custom():
    ups = [upd(1, "/toggle rsi"), upd(2, "/toggle rsi")]
    new, _, _, _ = apply_commands(ups, {"profile": "balanced"}, make_ledger(), CFG, "42")
    assert "toggles" not in new


def test_toggle_cannot_empty_the_list():
    s = {"profile": "balanced", "toggles": ["rsi"]}
    new, _, _, replies = apply_commands([upd(1, "/toggle rsi")], s, make_ledger(), CFG, "42")
    assert new["toggles"] == ["rsi"]
    assert "at least one indicator must stay on" in replies[0]


def test_profile_drops_custom_toggles():
    s = {"profile": "balanced", "toggles": ["rsi"]}
    new, _, _, _ = apply_commands([upd(1, "/profile balanced")], s, make_ledger(), CFG, "42")
    assert "toggles" not in new
