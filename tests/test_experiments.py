from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from models.registry import build_decoder
from run_experiments import load_experiment_matrix


def _base_config() -> dict:
    return {
        "experiment": {"name": "base", "seed": 42},
        "data": {},
        "tokenizer": {},
        "ssl": {"type": "wav2vec2"},
        "representation": {"type": "continuous"},
        "decoder": {"type": "linear"},
        "training": {"epochs": 1},
        "runtime": {"output_dir": "outputs/base"},
    }


def test_experiment_matrix_expands_named_overrides(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_path),
                "experiments": [
                    {
                        "name": "transformer",
                        "overrides": {
                            "decoder.type": "transformer",
                            "training.epochs": 3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    experiments = load_experiment_matrix(matrix_path)

    assert [item["name"] for item in experiments] == ["transformer"]
    assert experiments[0]["config"]["decoder"]["type"] == "transformer"
    assert experiments[0]["config"]["training"]["epochs"] == 3
    assert experiments[0]["config"]["experiment"]["name"] == "transformer"
    assert experiments[0]["config"]["runtime"]["output_dir"].endswith(
        "transformer"
    )


def test_experiment_matrix_rejects_duplicates(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_path),
                "experiments": [
                    {"name": "same", "overrides": {}},
                    {"name": "same", "overrides": {}},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_experiment_matrix(matrix_path)


def test_experiment_matrix_rejects_unimplemented_discrete_path(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_path),
                "experiments": [
                    {
                        "name": "discrete",
                        "overrides": {"representation.type": "kmeans"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continuous"):
        load_experiment_matrix(matrix_path)


def test_checked_experiment_matrix_contains_buildable_decoders() -> None:
    experiments = load_experiment_matrix("configs/experiments.yaml")

    for experiment in experiments:
        decoder = build_decoder(
            experiment["config"]["decoder"],
            input_dim=16,
            vocab_size=32,
        )
        assert decoder is not None
