"""Telegram notifier: message formatting, delivery and settings commands."""
from __future__ import annotations

import pandas as pd
import requests

from scanner.config import active_settings
from scanner.ledger import mark_took_it

FOOTER = "PAPER ONLY - not financial advice"
API = "https://api.telegram.org/bot{token}/{method}"


def _num(value, default=0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _qty(value) -> str:
    """Quantity rounded to 4 decimals, without trailing zeros (5 -> '5')."""
    q = _num(value)
    return f"{q:.4f}".rstrip("0").rstrip(".") or "0"


def format_weekly(picks: pd.DataFrame, settings: dict, sizing: list[dict],
                  skipped: dict[str, str], stops: list[float] | None = None) -> str:
    skipped = skipped or {}
    skip_lines = []
    if skipped:
        skip_lines.append("Skipped:")
        skip_lines += [f"- {t}: {r}" for t, r in skipped.items()]
    head = f"Profile: {settings['label']} | config {settings['version']}"
    if picks is None or len(picks) == 0:
        lines = ["No picks this week", head]
        lines += skip_lines
        lines.append(FOOTER)
        return "\n".join(lines)

    lines = ["Weekly picks", head, ""]
    for i, (ticker, row) in enumerate(picks.iterrows()):
        sz = sizing[i] if i < len(sizing) else {}
        lines.append(f"{i + 1}. {ticker} ({row.get('sector', '?')}) blend {_num(row['blend']):.1f}")
        lines.append(
            f"   technical {_num(row['technical']):.0f} | fundamental {_num(row['fundamental']):.0f}"
            f" | sentiment {_num(row['sentiment']):.0f} | risk {_num(row['risk']):.0f}"
        )
        stop = f"{stops[i]:.2f}" if stops is not None and i < len(stops) else "n/a"
        lines.append(f"   entry ref {_num(row['close']):.2f} | stop {stop}")
        lines.append(
            f"   qty {_qty(sz.get('quantity', 0))} | value {_num(sz.get('value')):.2f}"
            f" | risk {_num(sz.get('risk_amount')):.2f}"
        )
        reasons = list(row.get("reasons") or [])
        if reasons:
            lines.append("   why: " + "; ".join(str(r) for r in reasons))
        if not bool(row.get("earnings_known", True)):
            lines.append("   earnings date unknown")
        lines.append("")
    lines += skip_lines
    if skip_lines:
        lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines)


def _event_line(e: dict) -> str:
    kind, t = e.get("kind"), e.get("ticker")
    price = e.get("price")
    at = f" at {_num(price):.2f}" if price is not None else ""
    if kind == "stop":
        return f"Stop hit: {t}{at}"
    if kind == "sell_signal":
        return f"Sell signal: {t}{at}"
    if kind == "time_exit":
        return f"Time exit: {t}{at}"
    if kind == "closed":
        pnl = f", P&L {_num(e.get('pnl')):+.2f}" if e.get("pnl") is not None else ""
        return f"Closed: {t}{at} ({e.get('exit_reason', '')}){pnl}"
    if kind == "data_mismatch":
        return (f"ALERT: {t} price data looks inconsistent ({e.get('detail', '')}) - "
                f"check the position's price data")
    return f"{kind}: {t}"


def format_summary(summary: pd.DataFrame) -> list[str]:
    """Win/loss lines per config version (empty list when there are no closed trades)."""
    if summary is None or not len(summary):
        return []
    lines = ["Results by config version:"]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['config_version']}: {int(r['trades'])} trades, {int(r['wins'])} wins,"
            f" {int(r['losses'])} losses, win rate {float(r['win_rate']) * 100:.0f}%,"
            f" total P&L {float(r['total_pnl']):+.2f}"
        )
    return lines


def format_daily(events: list[dict], open_positions: pd.DataFrame, summary: pd.DataFrame,
                 positions: pd.DataFrame | None = None, account: dict | None = None) -> str:
    """`positions` (optional) shares the index of `open_positions` and holds
    last_close, unrealised and unrealised_pct for the rows it covers.
    `account` (optional) holds equity, peak and drawdown_pct."""
    lines = ["Daily update"]
    if account:
        lines.append(f"Paper equity {_num(account.get('equity')):.2f} "
                     f"(peak {_num(account.get('peak')):.2f}, "
                     f"drawdown {_num(account.get('drawdown_pct')):.1f}%)")
    lines.append("")
    if events:
        lines.append("Events:")
        lines += [f"- {_event_line(e)}" for e in events]
    else:
        lines.append("No changes today.")
    lines.append("")
    if open_positions is not None and len(open_positions):
        lines.append("Open positions:")
        for idx, r in open_positions.iterrows():
            line = (f"- {r['ticker']}: qty {_qty(r['quantity'])} | entry {_num(r['entry_price']):.2f}"
                    f" | stop {_num(r['stop']):.2f} | took it: {'yes' if r['took_it'] is True or r['took_it'] == True else 'no'}")
            if positions is not None and idx in positions.index:
                p = positions.loc[idx]
                line += (f" | last {_num(p['last_close']):.2f} | P&L {_num(p['unrealised']):+.2f}"
                         f" ({_num(p['unrealised_pct']):+.1f}%)")
            lines.append(line)
        lines.append("")
    summary_lines = format_summary(summary)
    if summary_lines:
        lines += summary_lines + [""]
    lines.append(FOOTER)
    return "\n".join(lines)


def split_message(text: str, limit: int = 4096) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    size = 0
    for line in text.split("\n"):
        pieces = [line[i:i + limit] for i in range(0, len(line), limit)] or [""]
        for piece in pieces:
            add = len(piece) + (1 if cur else 0)
            if cur and size + add > limit:
                parts.append("\n".join(cur))
                cur, size, add = [], 0, len(piece)
            cur.append(piece)
            size += add
    parts.append("\n".join(cur))
    return parts


class TelegramClient:
    def __init__(self, token: str, chat_id: str, session=None):
        self._token = token
        self.chat_id = str(chat_id)
        self.session = session if session is not None else requests.Session()

    def __repr__(self) -> str:
        return f"TelegramClient(chat_id={self.chat_id!r})"

    def _call(self, kind: str, method: str, **kw) -> dict:
        url = API.format(token=self._token, method=method)
        try:
            resp = getattr(self.session, kind)(url, timeout=30, **kw)
            data = resp.json()
        except Exception as exc:  # message may embed the URL/token: report type only
            raise RuntimeError(f"Telegram {method} request failed ({type(exc).__name__})") from None
        if not isinstance(data, dict) or not data.get("ok"):
            desc = data.get("description", "") if isinstance(data, dict) else ""
            desc = str(desc).replace(self._token, "***")
            raise RuntimeError(f"Telegram {method} returned an error: {desc}".rstrip(": "))
        return data

    def send(self, text: str) -> None:
        for part in split_message(text):
            self._call("post", "sendMessage", data={"chat_id": self.chat_id, "text": part})

    def get_updates(self, offset: int | None) -> list[dict]:
        params = {"timeout": 0}
        if offset is not None:
            params["offset"] = offset
        data = self._call("get", "getUpdates", params=params)
        return list(data.get("result", []))


_COMMANDS = {"profile": str.lower, "toggle": str.lower, "took": str.upper}


def parse_command(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    bits = text.strip().split()
    if len(bits) < 2 or not bits[0].startswith("/"):
        return None
    cmd = bits[0][1:].split("@")[0].lower()
    if cmd not in _COMMANDS:
        return None
    return {"cmd": cmd, "arg": _COMMANDS[cmd](bits[1])}


def apply_commands(updates, settings, ledger, cfg, chat_id):
    settings = dict(settings)
    if "toggles" in settings:
        settings["toggles"] = list(settings["toggles"])
    replies: list[str] = []
    ordered = sorted(updates, key=lambda u: u["update_id"])
    next_offset = max((u["update_id"] for u in ordered), default=None)
    next_offset = None if next_offset is None else next_offset + 1
    ledger = ledger.copy()

    for u in ordered:
        msg = u.get("message") or {}
        text = msg.get("text")
        if not text or str(msg.get("chat", {}).get("id")) != str(chat_id):
            continue
        cmd = parse_command(text)
        if cmd is None:
            continue
        arg = cmd["arg"]
        if cmd["cmd"] == "profile":
            if arg not in cfg["profiles"]:
                replies.append("unknown profile '" + arg + "'. valid profiles: "
                               + ", ".join(cfg["profiles"]))
                continue
            settings = {"profile": arg}
            replies.append(f"profile set to {active_settings(cfg, settings)['label']}")
        elif cmd["cmd"] == "toggle":
            if arg not in cfg["toggles"]:
                replies.append("unknown indicator '" + arg + "'. valid indicators: "
                               + ", ".join(cfg["toggles"]))
                continue
            preset = cfg["profiles"][settings.get("profile", "balanced")]["toggles"]
            current = list(settings.get("toggles", preset))
            new = [t for t in current if t != arg] if arg in current else current + [arg]
            if not new:
                replies.append("at least one indicator must stay on")
                continue
            new = [t for t in cfg["toggles"] if t in new]
            if set(new) == set(preset):
                settings.pop("toggles", None)
            else:
                settings["toggles"] = new
            state = "on" if arg in new else "off"
            replies.append(f"{arg} {state}; now {active_settings(cfg, settings)['label']}")
        else:
            rows = ledger[ledger["ticker"] == arg]
            if rows.empty:
                replies.append(f"no pick found for {arg}")
                continue
            ledger = mark_took_it(ledger, arg, rows["pick_date"].max())
            replies.append(f"marked {arg} as taken")
    return settings, ledger, next_offset, replies
