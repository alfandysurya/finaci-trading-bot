from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def cost_per_side(cfg: dict) -> float:
    """Effective one-way cost as a fraction of trade value (fees incl. VAT + slippage).

    The JFX/KBI fee is charged in full: its $0.10 cap only binds on orders above $200.
    """
    c = cfg["costs"]
    return (c["fee_rate"] + c["jfx_rate"]) * (1 + c["vat"]) + c["slippage_bps"] / 10_000
