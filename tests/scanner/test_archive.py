import hashlib

import pandas as pd
import pytest

from scanner.archive import append_manifest, write_snapshot

COLS = ["run_date", "file", "rows", "sha256", "fetched_at", "source", "archive_ok"]


def _frames():
    return {
        "prices": pd.DataFrame({"ticker": ["AAA", "BBB"], "close": [1.5, 2.5]}),
        "constituents": pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]}),
        "scores": pd.DataFrame({"score": [0.1]}),
        "mystery": pd.DataFrame({"x": [1]}),
    }


def test_snapshot_writes_dated_parquet_and_manifest_rows(tmp_path):
    rows = write_snapshot("2026-11-03", _frames(), tmp_path)
    by = {r["file"]: r for r in rows}
    assert set(by) == {
        "2026/11/03/prices.parquet",
        "2026/11/03/constituents.parquet",
        "2026/11/03/scores.parquet",
        "2026/11/03/mystery.parquet",
    }
    for f in by:
        assert (tmp_path / f).exists()
    assert by["2026/11/03/prices.parquet"]["rows"] == 2
    assert by["2026/11/03/prices.parquet"]["source"] == "yfinance"
    assert by["2026/11/03/constituents.parquet"]["source"] == "wikipedia"
    assert by["2026/11/03/scores.parquet"]["source"] == "scanner"
    assert by["2026/11/03/mystery.parquet"]["source"] == "unknown"
    for r in rows:
        assert r["run_date"] == "2026-11-03"
        assert r["fetched_at"]


def test_snapshot_never_overwrites(tmp_path):
    write_snapshot("2026-11-03", _frames(), tmp_path)
    p = tmp_path / "2026/11/03/prices.parquet"
    before = p.read_bytes()
    other = {"prices": pd.DataFrame({"ticker": ["ZZZ"], "close": [9.0]})}
    with pytest.raises(FileExistsError):
        write_snapshot("2026-11-03", other, tmp_path)
    assert p.read_bytes() == before


def test_overwrite_refusal_is_all_or_nothing(tmp_path):
    d = tmp_path / "2026/11/03"
    d.mkdir(parents=True)
    (d / "scores.parquet").write_bytes(b"original")
    frames = {k: v for k, v in _frames().items() if k != "mystery"}
    with pytest.raises(FileExistsError):
        write_snapshot("2026-11-03", frames, tmp_path)
    assert (d / "scores.parquet").read_bytes() == b"original"
    assert not (d / "prices.parquet").exists()
    assert not (d / "constituents.parquet").exists()


def test_manifest_checksum_matches_file(tmp_path):
    rows = write_snapshot("2026-11-03", _frames(), tmp_path)
    for r in rows:
        assert r["sha256"] == hashlib.sha256((tmp_path / r["file"]).read_bytes()).hexdigest()


def test_failed_run_marked_in_manifest(tmp_path):
    m = tmp_path / "sub" / "manifest.csv"
    rows = write_snapshot("2026-11-03", _frames(), tmp_path)
    append_manifest(rows, m, failed=True)
    df = pd.read_csv(m)
    assert list(df.columns) == COLS
    assert not df["archive_ok"].any()
    append_manifest([], m, failed=True, run_date="2026-11-04")
    df = pd.read_csv(m, keep_default_na=False)
    last = df.iloc[-1]
    assert last["run_date"] == "2026-11-04"
    assert last["file"] == "" and last["rows"] == 0 and last["sha256"] == ""
    assert str(last["archive_ok"]) == "False"


def test_invalid_run_date_rejected(tmp_path):
    for bad in ["2026-13-01", "20261103", "2026/11/03", "../x", ""]:
        with pytest.raises(ValueError):
            write_snapshot(bad, _frames(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_round_trip_preserves_index_and_dtypes(tmp_path):
    df = pd.DataFrame(
        {"close": [1.5, 2.5], "vol": [10, 20]},
        index=pd.Index(["AAA", "BBB"], name="ticker"),
    )
    write_snapshot("2026-11-03", {"prices": df}, tmp_path)
    back = pd.read_parquet(tmp_path / "2026/11/03/prices.parquet")
    assert list(back.columns) == ["ticker", "close", "vol"]
    assert back["close"].dtype == df["close"].dtype
    assert back["vol"].dtype == df["vol"].dtype
    assert list(back["ticker"]) == ["AAA", "BBB"]


def test_manifest_appends_without_duplicating_header(tmp_path):
    m = tmp_path / "manifest.csv"
    rows = write_snapshot("2026-11-03", _frames(), tmp_path)
    append_manifest(rows, m)
    append_manifest(rows, m)
    lines = m.read_text().splitlines()
    assert sum(1 for line in lines if line.startswith("run_date,")) == 1
    assert len(lines) == 1 + 2 * len(rows)
    df = pd.read_csv(m)
    assert df["archive_ok"].all()
