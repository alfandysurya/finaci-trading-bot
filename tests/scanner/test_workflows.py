"""Static checks of the GitHub Actions workflows and the setup guide (no network)."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WF_DIR = ROOT / ".github" / "workflows"
FILES = {"weekly": WF_DIR / "scanner-weekly.yml", "daily": WF_DIR / "scanner-daily.yml"}
README = ROOT / "scanner" / "README.md"
SECRETS = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "GOOGLE_SERVICE_ACCOUNT_JSON",
           "DRIVE_FOLDER_ID", "SHEET_ID"]
FOLDER_ID = "1WMlMlhfo9qGcnoAWESNs3WIp6bhv6qXH"


def load(name):
    return yaml.safe_load(FILES[name].read_text(encoding="utf-8"))


def triggers(wf):
    return wf.get("on", wf.get(True))


def cron(name):
    return triggers(load(name))["schedule"][0]["cron"]


def steps(wf):
    return next(iter(wf["jobs"].values()))["steps"]


def test_workflows_parse_and_reference_cli():
    for name in FILES:
        wf = load(name)
        on = triggers(wf)
        assert on["schedule"][0]["cron"]
        assert "workflow_dispatch" in on
        assert wf["permissions"] == {"contents": "write"}
        text = FILES[name].read_text(encoding="utf-8")
        assert f"uv run python -m scanner {name}" in text
        assert "uv sync --frozen" in text
        job = next(iter(wf["jobs"].values()))
        assert job["timeout-minutes"] == 30
        assert "actions/checkout@v4" in text and "astral-sh/setup-uv@" in text
    assert "as_of" in triggers(load("weekly"))["workflow_dispatch"]["inputs"]
    assert "--as-of" in FILES["weekly"].read_text(encoding="utf-8")


def test_daily_cron_is_after_us_close():
    minute, hour, dom, month, dow = cron("daily").split()
    assert int(hour) * 60 + int(minute) >= 21 * 60 + 30
    assert dow == "1-5"


def test_weekly_cron_is_sunday():
    assert cron("weekly") == "0 22 * * 0"


def test_workflows_share_concurrency_group():
    for name in FILES:
        c = load(name)["concurrency"]
        assert c["group"] == "scanner-state"
        assert c["cancel-in-progress"] is False


def test_workflows_only_commit_scanner_state():
    for name in FILES:
        text = FILES[name].read_text(encoding="utf-8")
        adds = re.findall(r"git add\s+(.*)", text)
        assert adds == ["scanner_state"]
        assert "commit -a" not in text and "-am" not in text
        assert "scanner: update state" in text
        commit = [s for s in steps(load(name)) if "git add" in s.get("run", "")]
        assert len(commit) == 1 and commit[0]["if"] == "always()"
        assert "github-actions[bot]" in commit[0]["run"]
        assert "41898282+github-actions[bot]@users.noreply.github.com" in commit[0]["run"]
        assert "data/archive" not in text


def test_no_literal_secrets():
    for name, path in FILES.items():
        text = path.read_text(encoding="utf-8")
        for secret in SECRETS:
            for line in text.splitlines():
                if secret in line:
                    assert re.search(rf"{secret}:\s*\$\{{\{{ secrets\.{secret} \}}\}}\s*$", line), line
        assert set(re.findall(r"secrets\.(\w+)", text)) == set(SECRETS)
        assert not re.search(r"\d{6,}:[A-Za-z0-9_-]{35}", text)
        assert "-----BEGIN" not in text and "private_key" not in text
        assert "echo" not in text


DRIVE_ID_LIKE = re.compile(
    r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{33}"
    r"(?![A-Za-z0-9_-])")


def test_readme_mentions_all_secret_names_and_folder_id():
    """F-7: the README names every secret but never embeds the real Drive folder id."""
    text = README.read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret in text
    assert "<your folder id>" in text
    assert "drive.google.com/drive/folders/<your folder id>" in text
    assert FOLDER_ID not in text
    assert not DRIVE_ID_LIKE.search(text), DRIVE_ID_LIKE.search(text)
    assert not re.search(r"\d{6,}:[A-Za-z0-9_-]{35}", text)
    assert "-----BEGIN" not in text


def test_drive_id_pattern_catches_real_ids():
    assert DRIVE_ID_LIKE.search(f"id is {FOLDER_ID}.")
    assert not DRIVE_ID_LIKE.search("GOOGLE_SERVICE_ACCOUNT_JSON and DRIVE_FOLDER_ID")


@pytest.mark.parametrize("name", ["daily", "weekly"])
def test_workflows_have_market_hours_guard_before_scan(name):
    """C-1: weekly has the same guard as daily; Sunday (dow 7) stays allowed."""
    st = steps(load(name))
    ids = [s.get("id") for s in st]
    assert "market-guard" in ids
    guard = st[ids.index("market-guard")]
    scan = next(s for s in st if f"python -m scanner {name}" in s.get("run", ""))
    assert st.index(guard) < st.index(scan)
    run = guard["run"]
    assert "America/New_York" in run and "930" in run and "1615" in run
    assert '"$dow" -le 5' in run  # weekdays only: the Sunday weekly run is allowed
    assert "Refusing to run: US market is open; run after 16:00 America/New_York" in run
    assert "exit 1" in run
    assert "if" not in scan or "always" not in str(scan["if"])


def test_guard_steps_are_identical():
    def guard(name):
        st = steps(load(name))
        return next(s for s in st if s.get("id") == "market-guard")["run"]
    assert guard("weekly") == guard("daily")


def test_gitignore_excludes_state_tmp_files():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "scanner_state/*.tmp" in lines


def test_readme_covers_setup_notes_and_known_limits():
    """C-4 / C-5 wording and notes."""
    text = README.read_text(encoding="utf-8")
    assert "top 3 picks" in text
    step7 = text.split("### 7.")[1].split("\n### ")[0]
    assert "GOOGLE_SERVICE_ACCOUNT_JSON" in step7 and "DRIVE_FOLDER_ID" in step7
    for other in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SHEET_ID", "five"):
        assert other not in step7
    step8 = text.split("### 8.")[1].split("\n### ")[0]
    assert "Yahoo" in step8 and "Wikipedia" in step8
    limits = text.split("## Known limits")[1]
    for phrase in ("20%", "reset", "60 days", "config_version", "sizing:", "gates:",
                   "next open", "dividend", "rejects", "scores"):
        assert phrase in limits, phrase
    for phrase in ("check-drive", "shared drive", "Read and write permissions",
                   "branch protection"):
        assert phrase in text, phrase


def test_readme_reset_names_state_json_and_command_timing():
    text = README.read_text(encoding="utf-8")
    reset = text.split("## Reset")[1].split("\n## ")[0]
    assert "scanner_state/ledger.csv" in reset and "scanner_state/state.json" in reset
    assert "peak" in reset
    for keep in ("settings.json", "telegram_offset.json", "manifest.csv"):
        assert keep in reset
    assert "next run" in text and "weekend" in text
