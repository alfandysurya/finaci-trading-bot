"""Append-only, point-in-time archive of everything the scanner saw each run."""
from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCES = {
    "prices": "yfinance",
    "fundamentals": "yfinance",
    "analyst": "yfinance",
    "earnings": "yfinance",
    "constituents": "wikipedia",
    "scores": "scanner",
}

MANIFEST_COLUMNS = ["run_date", "file", "rows", "sha256", "fetched_at", "source", "archive_ok"]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_date(run_date: str) -> datetime:
    if not isinstance(run_date, str) or not _DATE_RE.match(run_date):
        raise ValueError(f"run_date must be YYYY-MM-DD, got {run_date!r}")
    try:
        return datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"invalid run_date {run_date!r}") from exc


def write_snapshot(run_date: str, frames: dict[str, pd.DataFrame], root: Path) -> list[dict]:
    """Write one parquet per frame; an unnamed non-default index comes back as an "index" column."""
    d = _parse_date(run_date)
    root = Path(root)
    day_dir = root / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
    for name in frames:
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid frame name {name!r}")
    payloads: dict[str, bytes] = {}
    for name, frame in frames.items():
        if isinstance(frame.index, pd.RangeIndex) and frame.index.name is None:
            out = frame
        else:
            out = frame.reset_index()
        buf = io.BytesIO()
        out.to_parquet(buf)
        payloads[name] = buf.getvalue()

    targets = {name: day_dir / f"{name}.parquet" for name in frames}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing:
        raise FileExistsError(f"archive files already exist: {existing}")

    day_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, frame in frames.items():
        path = targets[name]
        data = payloads[name]
        with open(path, "xb") as fh:
            fh.write(data)
        rows.append(
            {
                "run_date": run_date,
                "file": path.relative_to(root).as_posix(),
                "rows": len(frame),
                "sha256": hashlib.sha256(data).hexdigest(),
                "fetched_at": _now(),
                "source": SOURCES.get(name, "unknown"),
            }
        )
    return rows


def append_manifest(
    rows: list[dict],
    manifest_path: Path,
    failed: bool = False,
    run_date: str | None = None,
) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and failed:
        rows = [
            {
                "run_date": run_date or "",
                "file": "",
                "rows": 0,
                "sha256": "",
                "fetched_at": _now(),
                "source": "",
            }
        ]
    new = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({**{c: r.get(c, "") for c in MANIFEST_COLUMNS[:-1]}, "archive_ok": not failed})
