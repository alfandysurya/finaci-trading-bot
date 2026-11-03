"""Static checks of the GitHub Actions workflows and the setup guide (no network)."""
import re
from pathlib import Path

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


def test_readme_mentions_all_secret_names_and_folder_id():
    text = README.read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret in text
    assert FOLDER_ID in text
    assert not re.search(r"\d{6,}:[A-Za-z0-9_-]{35}", text)
    assert "-----BEGIN" not in text
