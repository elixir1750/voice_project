from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REQUIRED_SECTIONS = {
    "experiment",
    "data",
    "tokenizer",
    "ssl",
    "representation",
    "decoder",
    "training",
    "runtime",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_SECTIONS.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    non_mappings = sorted(
        section for section in REQUIRED_SECTIONS if not isinstance(config[section], Mapping)
    )
    if non_mappings:
        raise ValueError(
            f"Configuration sections must be mappings: {', '.join(non_mappings)}"
        )


def apply_overrides(
    config: dict[str, Any],
    overrides: Sequence[str] | None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    for override in overrides or ():
        if "=" not in override:
            raise ValueError(f"Override must use KEY=VALUE syntax: {override}")
        dotted_key, raw_value = override.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ValueError(f"Override must use KEY=VALUE syntax: {override}")

        target: dict[str, Any] = updated
        for key in keys[:-1]:
            existing = target.get(key)
            if existing is None:
                target[key] = {}
            elif not isinstance(existing, dict):
                raise ValueError(f"Cannot assign nested override below: {key}")
            target = target[key]
        target[keys[-1]] = yaml.safe_load(raw_value)
    validate_config(updated)
    return updated


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)
