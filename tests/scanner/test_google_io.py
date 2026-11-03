"""Google Drive / Sheets helpers, tested against fake services (no network)."""
from __future__ import annotations

import json
import re

import httplib2
import numpy as np
import pandas as pd
import pytest
from googleapiclient.errors import HttpError

from scanner import google_io

FOLDER = "ROOT"
_NAME_RE = r"name = '((?:[^'\\]|\\.)*)'"
_PARENT_RE = r"'([^']*)' in parents"


class _Req:
    def __init__(self, fn):
        self._fn = fn

    def execute(self):
        return self._fn()


class FakeDrive:
    def __init__(self, create_error=None):
        self.items = {}  # id -> dict(name, parent, mimeType)
        self.queries = []
        self.deleted = []
        self.uploaded = []
        self.create_error = create_error
        self._n = 0

    def _add(self, name, parent, mime):
        self._n += 1
        fid = f"id{self._n}"
        self.items[fid] = {"name": name, "parent": parent, "mimeType": mime}
        return fid

    def files(self):
        return self

    def list(self, **kw):
        q = kw["q"]
        self.queries.append(q)
        assert kw.get("supportsAllDrives") is True
        assert kw.get("includeItemsFromAllDrives") is True
        assert "trashed = false" in q
        name = re.search(_NAME_RE, q).group(1).replace("\\'", "'").replace("\\\\", "\\")
        parent = re.search(_PARENT_RE, q).group(1)
        want_folder = "mimeType = 'application/vnd.google-apps.folder'" in q
        out = [
            {"id": i, "name": v["name"]}
            for i, v in self.items.items()
            if v["name"] == name
            and v["parent"] == parent
            and (v["mimeType"] == google_io.FOLDER_MIME) == want_folder
        ]
        return _Req(lambda: {"files": out})

    def create(self, body=None, media_body=None, fields=None, supportsAllDrives=None):
        def run():
            if self.create_error:
                raise self.create_error
            assert supportsAllDrives is True
            mime = body.get("mimeType", "file")
            fid = self._add(body["name"], body["parents"][0], mime)
            if media_body is not None:
                self.uploaded.append((body["name"], body["parents"][0], media_body.mimetype()))
            return {"id": fid}

        return _Req(run)

    def delete(self, fileId=None, supportsAllDrives=None):
        def run():
            self.deleted.append(fileId)
            del self.items[fileId]

        return _Req(run)

    def names(self):
        return [v["name"] for v in self.items.values()]


def _make_files(tmp_path, rels):
    for r in rels:
        p = tmp_path / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"data")


def test_upload_skips_existing_names(tmp_path):
    _make_files(tmp_path, ["2026/11/03/prices.parquet", "2026/11/03/scores.parquet"])
    drive = FakeDrive()
    y = drive._add("2026", FOLDER, google_io.FOLDER_MIME)
    m = drive._add("11", y, google_io.FOLDER_MIME)
    d = drive._add("03", m, google_io.FOLDER_MIME)
    drive._add("prices.parquet", d, "application/octet-stream")
    google_io.upload_to_drive(
        drive, tmp_path, ["2026/11/03/prices.parquet", "2026/11/03/scores.parquet"], FOLDER
    )
    assert [u[0] for u in drive.uploaded] == ["scores.parquet"]
    assert drive.deleted == []
    assert drive.uploaded[0][2] == "application/octet-stream"


def test_upload_creates_dated_subfolders(tmp_path):
    _make_files(tmp_path, ["2026/11/03/prices.parquet", "2026/11/03/scores.parquet"])
    drive = FakeDrive()
    google_io.upload_to_drive(
        drive, tmp_path, ["2026/11/03/prices.parquet", "2026/11/03/scores.parquet"], FOLDER
    )
    folders = [v for v in drive.items.values() if v["mimeType"] == google_io.FOLDER_MIME]
    assert [f["name"] for f in folders] == ["2026", "11", "03"]  # created once, reused
    assert folders[0]["parent"] == FOLDER
    assert sorted(u[0] for u in drive.uploaded) == ["prices.parquet", "scores.parquet"]
    day_id = [i for i, v in drive.items.items() if v["name"] == "03"][0]
    assert all(u[1] == day_id for u in drive.uploaded)


def test_upload_rejects_path_traversal(tmp_path):
    drive = FakeDrive()
    for bad in ["../x.txt", "2026/../../x.txt", "/abs/x.txt", "C:/abs/x.txt", "\\abs\\x.txt"]:
        with pytest.raises(ValueError):
            google_io.upload_to_drive(drive, tmp_path, [bad], FOLDER)
    assert drive.items == {}


def test_upload_escapes_quotes_in_names(tmp_path):
    _make_files(tmp_path, ["o'brien/it's.txt"])
    drive = FakeDrive()
    google_io.upload_to_drive(drive, tmp_path, ["o'brien/it's.txt"], FOLDER)
    assert any("o\\'brien" in q for q in drive.queries)
    assert any("it\\'s.txt" in q for q in drive.queries)
    assert "it's.txt" in drive.names()


# ---------------------------------------------------------------- sheets
class FakeSheets:
    def __init__(self, tabs=("Notes", "Picks")):
        self.tabs = list(tabs)
        self.log = []

    def spreadsheets(self):
        return self

    def get(self, spreadsheetId=None, fields=None):
        return _Req(lambda: {"sheets": [{"properties": {"title": t}} for t in self.tabs]})

    def batchUpdate(self, spreadsheetId=None, body=None):
        def run():
            for r in body["requests"]:
                t = r["addSheet"]["properties"]["title"]
                self.tabs.append(t)
                self.log.append(("addSheet", t))
            return {}

        return _Req(run)

    def values(self):
        return self

    def clear(self, spreadsheetId=None, range=None, body=None):
        return _Req(lambda: self.log.append(("clear", range)) or {})

    def update(self, spreadsheetId=None, range=None, valueInputOption=None, body=None):
        def run():
            self.log.append(("update", range, valueInputOption, body["values"]))
            return {}

        return _Req(run)


def test_mirror_only_touches_listed_tabs():
    sh = FakeSheets()
    google_io.mirror_to_sheets(sh, "S", {"Picks": pd.DataFrame({"a": [1], "b": ["x"]})})
    assert sh.log == [
        ("clear", "'Picks'"),
        ("update", "'Picks'!A1", "RAW", [["a", "b"], [1, "x"]]),
    ]
    assert all("Notes" not in str(e) for e in sh.log)


def test_mirror_creates_missing_tab():
    sh = FakeSheets(tabs=("Notes",))
    google_io.mirror_to_sheets(sh, "S", {"Open": pd.DataFrame({"a": [1]})})
    assert sh.log[0] == ("addSheet", "Open")
    assert ("clear", "'Open'") in sh.log


def test_mirror_cells_are_json_safe():
    df = pd.DataFrame(
        {
            "n": [np.nan, 2],
            "t": [pd.NaT, pd.Timestamp("2026-11-03 10:00")],
            "i": np.array([5, 6], dtype="int64"),
            "l": [[1, 2], None],
            "f": np.array([1.5, 2.5], dtype="float64"),
        }
    )
    sh = FakeSheets()
    google_io.mirror_to_sheets(sh, "S", {"Picks": df})
    values = sh.log[-1][3]
    json.dumps(values)  # must serialise
    assert values[1] == ["", "", 5, "[1, 2]", 1.5]
    assert values[2][0] == 2
    assert values[2][1].startswith("2026-11-03T10:00")
    assert values[2][3] == ""
    assert type(values[1][2]) is int and type(values[1][4]) is float


def test_mirror_empty_frame_writes_header():
    sh = FakeSheets()
    google_io.mirror_to_sheets(sh, "S", {"Picks": pd.DataFrame(columns=["a", "b"])})
    assert sh.log[-1][3] == [["a", "b"]]


# ---------------------------------------------------------------- misc
def test_build_services_invalid_json_message_has_no_content():
    secret = "SUPERSECRET{not json"
    with pytest.raises(ValueError) as e:
        google_io.build_services(secret)
    assert "GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON" in str(e.value)
    assert "SUPERSECRET" not in str(e.value) and "SUPERSECRET" not in repr(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


def _http_error(status, reason):
    body = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
    return HttpError(resp=httplib2.Response({"status": str(status)}), content=body)


def test_check_drive_write_raises_on_quota_error():
    drive = FakeDrive(create_error=_http_error(403, "storageQuotaExceeded"))
    with pytest.raises(RuntimeError) as e:
        google_io.check_drive_write(drive, FOLDER)
    msg = str(e.value)
    assert "quota" in msg.lower()
    assert "shared drive" in msg


def test_check_drive_write_raises_on_permission_error():
    drive = FakeDrive(create_error=_http_error(404, "notFound"))
    with pytest.raises(RuntimeError, match="not shared with the service account as an editor"):
        google_io.check_drive_write(drive, FOLDER)


def test_check_drive_write_deletes_test_file_and_returns_id():
    drive = FakeDrive()
    fid = google_io.check_drive_write(drive, FOLDER)
    assert drive.deleted == [fid]
    assert drive.items == {}
    assert fid.startswith("id")
