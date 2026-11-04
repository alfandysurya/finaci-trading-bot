from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_scanner_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "scanner.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def config_version(profile: str, toggles: list[str], weights: dict[str, float]) -> str:
    """Short hash of (profile, sorted enabled toggles, weights); 8 hex chars."""
    canon = json.dumps(
        {"profile": profile, "toggles": sorted(toggles), "weights": weights},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(canon.encode()).hexdigest()[:8]


def active_settings(cfg: dict, settings: dict) -> dict:
    profile = settings.get("profile", "balanced")
    p = cfg["profiles"][profile]
    toggles = settings.get("toggles")
    toggles = list(p["toggles"] if toggles is None else toggles)
    weights = dict(p["weights"])
    label = p["name"]
    if set(toggles) != set(p["toggles"]):
        label = f"Custom (based on {label})"
    return {
        "profile": profile,
        "toggles": toggles,
        "weights": weights,
        "version": config_version(profile, toggles, weights),
        "label": label,
    }
