"""Command line: python -m scanner weekly|daily|check-drive. PAPER ONLY.

Secrets come from environment variables only and are never printed.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd

from scanner.config import ROOT, load_scanner_config
from scanner.pipeline import default_deps, run_daily, run_weekly, scrub, secret_values

DRYRUN_DIR = ".dryrun"  # dry runs never touch the real scanner_state/ or data/archive


def _date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from None
    return value


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scanner",
                                description="Weekly stock scanner (paper only).")
    p.add_argument("command", choices=["weekly", "daily", "check-drive"])
    p.add_argument("--as-of", type=_date, default=None,
                   help="run date YYYY-MM-DD (default: today, UTC)")
    p.add_argument("--dry-run", action="store_true",
                   help="print messages instead of sending; skip Google Drive and Sheets")
    p.add_argument("--state-dir", default=None,
                   help="state directory (default: scanner_state/ in the repo; "
                        ".dryrun/state with --dry-run)")
    return p.parse_args(argv)


def resolve_as_of(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))


def _check_drive(env: Mapping[str, str]) -> str:
    from scanner.google_io import build_services, check_drive_write
    from scanner.pipeline import ENV_DRIVE, ENV_SA

    missing = [k for k in (ENV_SA, ENV_DRIVE) if not env.get(k)]
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))
    drive, _ = build_services(env[ENV_SA])
    return check_drive_write(drive, env[ENV_DRIVE])


def resolve_paths(args: argparse.Namespace, cfg: dict) -> tuple[Path, Path]:
    """(state_dir, archive_root). A dry run defaults both under ROOT/.dryrun; an
    explicit --state-dir still wins for state."""
    if args.dry_run:
        state_default = ROOT / DRYRUN_DIR / "state"
        archive_root = ROOT / DRYRUN_DIR / "archive"
    else:
        state_default = ROOT / cfg["archive"]["state_dir"]
        archive_root = ROOT / cfg["archive"]["archive_dir"]
    state_dir = Path(args.state_dir) if args.state_dir else state_default
    return state_dir, archive_root


def main(argv=None, env: Mapping[str, str] | None = None) -> int:
    args = parse_args(argv)
    env = os.environ if env is None else env

    if args.command == "check-drive":
        try:
            file_id = _check_drive(env)
        except Exception as exc:
            print(scrub(f"check-drive failed: {exc}", secret_values(env)))
            return 1
        print(f"Drive write OK: test file {file_id} created and deleted")
        return 0

    cfg = load_scanner_config()
    state_dir, archive_root = resolve_paths(args, cfg)
    deps = default_deps(env, dry_run=args.dry_run)
    as_of = resolve_as_of(args.as_of)
    runner = run_weekly if args.command == "weekly" else run_daily
    result = runner(cfg, deps, as_of, state_dir, archive_root=archive_root)

    print(f"{args.command} run for {as_of:%Y-%m-%d} finished")
    if result.get("skipped_reason"):
        print(f"skipped: {result['skipped_reason']}")
    picks = result.get("picks")
    if picks is not None and len(picks):
        print("picks: " + ", ".join(map(str, picks.index)))
    print(f"messages sent: {len(result['sent'])}; archive ok: {result['archive_ok']}")
    for err in result["errors"]:
        print(f"error: {err}")
    if result.get("send_failed"):
        # All state is already saved (the message is queued in state.json for the next
        # run); the workflow's commit step runs with if: always().
        print("error: a weekly/daily message was not delivered; it will be re-sent next run")
        return 1
    # Other non-fatal errors (already alerted) exit 0.
    return 0
