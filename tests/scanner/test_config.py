from scanner.config import active_settings, config_version, load_scanner_config

TOGGLES = ["sma_trend", "rsi", "macd", "volume", "valuation_pe", "ev_ebitda", "fcf_yield", "peg",
           "eps_growth", "forward_pe", "roe", "debt_equity", "analyst"]
W = {"technical": 0.3, "fundamental": 0.3, "sentiment": 0.2, "risk": 0.2}


def test_config_version_stable_and_order_independent():
    a = config_version("balanced", TOGGLES, W)
    b = config_version("balanced", list(reversed(TOGGLES)), dict(reversed(list(W.items()))))
    assert a == b and len(a) == 8


def test_config_version_changes_with_toggle():
    assert config_version("balanced", TOGGLES, W) != config_version("balanced", TOGGLES[:-1], W)


def test_label_custom_when_toggles_differ_from_profile_preset():
    cfg = load_scanner_config()
    preset = active_settings(cfg, {"profile": "balanced"})
    assert preset["label"] == "Balanced"
    preset_toggles = cfg["profiles"]["balanced"]["toggles"]
    shuffled = active_settings(cfg, {"profile": "balanced", "toggles": list(reversed(preset_toggles))})
    assert shuffled["label"] == "Balanced"
    assert shuffled["version"] == preset["version"]
    custom = active_settings(cfg, {"profile": "balanced", "toggles": preset_toggles[:-1]})
    assert custom["label"] == "Custom (based on Balanced)"
    assert custom["version"] != preset["version"]
    assert custom["weights"] == cfg["profiles"]["balanced"]["weights"]


def test_profile_weights_sum_to_one():
    cfg = load_scanner_config()
    assert set(cfg["profiles"]) == {"balanced", "value", "momentum"}
    for p in cfg["profiles"].values():
        assert abs(sum(p["weights"].values()) - 1.0) < 1e-9
        assert set(p["toggles"]) <= set(TOGGLES)
