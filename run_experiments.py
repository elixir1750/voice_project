from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from config import apply_overrides, load_config


def _override_arguments(overrides: Mapping[str, Any]) -> list[str]:
    return [
        f"{key}={yaml.safe_dump(value, default_flow_style=True).strip()}"
        for key, value in overrides.items()
    ]


def load_experiment_matrix(
    path: str | Path,
) -> list[dict[str, Any]]:
    matrix_path = Path(path)
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Experiment matrix must be a mapping")
    base_value = payload.get("base_config")
    if not isinstance(base_value, str) or not base_value:
        raise ValueError("Experiment matrix requires base_config")
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = matrix_path.parent / base_path
    base_config = load_config(base_path)
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("Experiment matrix requires a non-empty experiments list")

    seen: set[str] = set()
    experiments: list[dict[str, Any]] = []
    base_output_dir = Path(str(base_config["runtime"]["output_dir"]))
    for item in raw_experiments:
        if not isinstance(item, Mapping):
            raise ValueError("Each experiment must be a mapping")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each experiment requires a name")
        if name in seen:
            raise ValueError(f"Duplicate experiment name: {name}")
        seen.add(name)
        overrides = item.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Overrides for {name} must be a mapping")
        config = apply_overrides(
            base_config,
            _override_arguments(overrides),
        )
        config["experiment"]["name"] = name
        if "runtime.output_dir" not in overrides:
            config["runtime"]["output_dir"] = str(base_output_dir.parent / name)
        representation_type = str(config["representation"].get("type", "")).lower()
        if representation_type not in {"continuous", "kmeans"}:
            raise ValueError(
                f"Experiment {name} must use the implemented continuous "
                "or kmeans representation"
            )
        experiments.append(
            {
                "name": name,
                "overrides": dict(overrides),
                "config": config,
            }
        )
    return experiments


def run_experiment_matrix(
    matrix_path: str | Path,
    execute: bool = False,
    names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    experiments = load_experiment_matrix(matrix_path)
    selected_names = set(names or ())
    if selected_names:
        known_names = {item["name"] for item in experiments}
        unknown = sorted(selected_names - known_names)
        if unknown:
            raise ValueError(f"Unknown experiments: {', '.join(unknown)}")
        experiments = [
            item for item in experiments if item["name"] in selected_names
        ]

    if not execute:
        preview = [
            {
                "name": item["name"],
                "overrides": item["overrides"],
                "output_dir": item["config"]["runtime"]["output_dir"],
            }
            for item in experiments
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return preview

    from train import fit

    results = []
    for item in experiments:
        training_result = fit(item["config"])
        results.append(
            {
                "name": item["name"],
                **training_result,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results
