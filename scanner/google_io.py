"""Google Drive upload (append-only archive) and one-way Sheets mirror.

Both run as a service account. Nothing here ever echoes credential content.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from googleapiclient.errors import HttpError

FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]
TEST_FILE_NAME = "_sa_write_test.txt"
_BAD_JSON = "GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON"


def build_services(service_account_json: str) -> tuple[Any, Any]:
    """Return (drive v3, sheets v4) clients for the given service account JSON."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    try:
        info = json.loads(service_account_json)
        if not isinstance(info, dict):
            raise ValueError
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        # Deliberately drop the original exception: it may quote the secret.
        raise ValueError(_BAD_JSON) from None
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive, sheets


# ------------------------------------------------------------------ Drive
def _q(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _find(drive: Any, name: str, parent: str, folder: bool) -> str | None:
    clauses = [f"name = '{_q(name)}'", f"'{_q(parent)}' in parents", "trashed = false"]
    if folder:
        clauses.append(f"mimeType = '{FOLDER_MIME}'")
    else:
        clauses.append(f"mimeType != '{FOLDER_MIME}'")
    resp = drive.files().list(
        q=" and ".join(clauses),
        fields="files(id, name)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _ensure_folder(drive: Any, name: str, parent: str) -> str:
    found = _find(drive, name, parent, folder=True)
    if found:
        return found
    created = drive.files().create(
        body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent]},
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def _validate_rel(rel: str) -> list[str]:
    norm = rel.replace("\\", "/")
    p = PurePosixPath(norm)
    if (
        not norm
        or p.is_absolute()
        or (len(norm) > 1 and norm[1] == ":")
        or norm.endswith("/")
        or ".." in p.parts
    ):
        raise ValueError(f"unsafe archive path: {rel!r}")
    parts = [s for s in p.parts if s not in ("", ".")]
    if not parts:
        raise ValueError(f"unsafe archive path: {rel!r}")
    return parts


def upload_to_drive(drive: Any, local_root: Path, rel_paths: list[str], folder_id: str) -> None:
    """Upload files under dated subfolders of folder_id. Never overwrites or deletes."""
    from googleapiclient.http import MediaFileUpload

    # Validate everything first so a bad path uploads nothing.
    parts_list = [_validate_rel(r) for r in rel_paths]
    folder_cache: dict[tuple[str, ...], str] = {}
    for rel, parts in zip(rel_paths, parts_list):
        parent = folder_id
        for i in range(len(parts) - 1):
            key = tuple(parts[: i + 1])
            if key not in folder_cache:
                folder_cache[key] = _ensure_folder(drive, parts[i], parent)
            parent = folder_cache[key]
        name = parts[-1]
        if _find(drive, name, parent, folder=False):
            continue
        media = MediaFileUpload(
            str(Path(local_root) / Path(*parts)),
            mimetype="application/octet-stream",
            resumable=False,
        )
        drive.files().create(
            body={"name": name, "parents": [parent]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()


def _reasons(err: HttpError) -> str:
    try:
        data = json.loads(err.content.decode("utf-8", "replace"))
        errs = data.get("error", {}).get("errors", [])
        return " ".join(str(e.get("reason", "")) for e in errs)
    except Exception:
        return ""


def check_drive_write(drive: Any, folder_id: str) -> str:
    """Write and delete a tiny test file in the folder; return the created file id."""
    import io

    from googleapiclient.http import MediaIoBaseUpload

    media = MediaIoBaseUpload(
        io.BytesIO(b"service account write test"), mimetype="text/plain", resumable=False
    )
    try:
        created = drive.files().create(
            body={"name": TEST_FILE_NAME, "parents": [folder_id]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        file_id = created["id"]
    except HttpError as err:
        status = getattr(err.resp, "status", None)
        reasons = _reasons(err)
        if "storageQuotaExceeded" in reasons:
            raise RuntimeError(
                "Drive write failed: service account storage quota exceeded "
                "(service accounts have no storage of their own). Fix: use a folder in "
                "your own Drive shared with the service account, or a shared drive."
            ) from None
        if status in (403, 404):
            raise RuntimeError(
                "Drive write failed: the folder is not shared with the service account "
                "as an editor (or the folder id is wrong)."
            ) from None
        raise RuntimeError(f"Drive write failed with HTTP status {status}.") from None
    try:
        drive.files().delete(fileId=file_id, supportsAllDrives=True).execute()
    except Exception:
        raise RuntimeError(
            "Write test succeeded but the test file could not be deleted; "
            f"remove file id {file_id} from the folder by hand."
        ) from None
    return file_id


# ----------------------------------------------------------------- Sheets
def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, dict, set, np.ndarray)):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, set):
            v = sorted(v, key=str)
        return json.dumps(v, default=_json_default)
    if isinstance(v, pd.Timestamp):
        return "" if v is pd.NaT else v.isoformat()
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return ""
    if isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def mirror_to_sheets(sheets: Any, sheet_id: str, tabs: dict[str, pd.DataFrame]) -> None:
    """Clear and rewrite only the listed tabs; any other tab is left alone."""
    meta = sheets.spreadsheets().get(
        spreadsheetId=sheet_id, fields="sheets.properties.title"
    ).execute()
    existing = {s["properties"]["title"] for s in meta.get("sheets", [])}
    for name, df in tabs.items():
        if name not in existing:
            sheets.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body={"requests": [{"addSheet": {"properties": {"title": name}}}]},
            ).execute()
            existing.add(name)
        rng = "'" + name.replace("'", "''") + "'"
        sheets.spreadsheets().values().clear(
            spreadsheetId=sheet_id, range=rng, body={}
        ).execute()
        rows = [[str(c) for c in df.columns]]
        rows += [[_cell(v) for v in rec] for rec in df.itertuples(index=False, name=None)]
        sheets.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"{rng}!A1",
            valueInputOption="RAW",
            body={"values": rows},
        ).execute()
