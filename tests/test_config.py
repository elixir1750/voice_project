from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import apply_overrides, load_config, validate_config


def _minimal_config() -> dict:
    return {
        "experiment": {"name": "test"},
        "data": {},
        "tokenizer": {},
        "ssl": {},
        "representation": {},
        "decoder": {},
        "training": {"epochs": 1},
        "runtime": {"device": "auto"},
    }


def test_load_config_and_apply_typed_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_minimal_config()), encoding="utf-8")

    config = load_config(path)
    updated = apply_overrides(
        config,
        ["training.epochs=3", "runtime.device=cpu", "training.amp=false"],
    )

    assert updated["training"]["epochs"] == 3
    assert updated["runtime"]["device"] == "cpu"
    assert updated["training"]["amp"] is False
    assert config["training"]["epochs"] == 1


def test_validate_config_reports_missing_top_level_section() -> None:
    config = _minimal_config()
    del config["ssl"]

    with pytest.raises(ValueError, match="ssl"):
        validate_config(config)


def test_override_rejects_invalid_assignment() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_overrides(_minimal_config(), ["training.epochs"])
