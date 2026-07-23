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


def test_experiment_matrix_rejects_unknown_representation_path(
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
                        "overrides": {"representation.type": "vq"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continuous or kmeans"):
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


def test_research_a_decoder_experiments_are_anchored_to_layer_9() -> None:
    experiments = {
        item["name"]: item["config"]
        for item in load_experiment_matrix("configs/experiments.yaml")
    }

    assert {
        "decoder_linear_layer9",
        "decoder_mlp_layer9",
        "decoder_transformer_layer9",
    }.issubset(experiments)
    assert experiments["decoder_linear_layer9"]["decoder"]["type"] == "linear"
    assert experiments["decoder_mlp_layer9"]["decoder"]["type"] == "mlp"
    assert (
        experiments["decoder_transformer_layer9"]["decoder"]["type"]
        == "transformer"
    )
    for name in (
        "decoder_linear_layer9",
        "decoder_mlp_layer9",
        "decoder_transformer_layer9",
    ):
        assert experiments[name]["ssl"]["layer"] == 9
        assert experiments[name]["data"]["train_samples"] == 3600
        assert experiments[name]["training"]["save_checkpoints"] is False


def test_research_d_data_scale_experiments_use_layer_9_mlp() -> None:
    experiments = {
        item["name"]: item["config"]
        for item in load_experiment_matrix("configs/experiments.yaml")
    }
    expected_samples = {
        "train_samples_900_layer9": 900,
        "train_samples_1800_layer9": 1800,
        "train_samples_3600_layer9": 3600,
        "train_samples_5400_layer9": 5400,
        "train_samples_7200_layer9": 7200,
    }

    assert expected_samples.keys() <= experiments.keys()
    for name, sample_count in expected_samples.items():
        config = experiments[name]
        assert config["ssl"]["layer"] == 9
        assert config["representation"]["type"] == "continuous"
        assert config["decoder"]["type"] == "mlp"
        assert config["data"]["train_samples"] == sample_count


def test_research_b_kmeans_experiments_use_layer_9_mlp() -> None:
    experiments = {
        item["name"]: item["config"]
        for item in load_experiment_matrix("configs/experiments.yaml")
    }
    expected = {
        "kmeans_k100": (100, False),
        "kmeans_k500": (500, False),
        "kmeans_k1000": (1000, False),
        "kmeans_k500_dedup": (500, True),
    }

    assert expected.keys() <= experiments.keys()
    for name, (codebook_size, dedup) in expected.items():
        config = experiments[name]
        assert config["ssl"]["layer"] == 9
        assert config["data"]["train_samples"] == 3600
        assert config["training"]["epochs"] == 10
        assert config["decoder"]["type"] == "mlp"
        assert config["representation"]["type"] == "kmeans"
        assert config["representation"]["codebook_size"] == codebook_size
        assert config["representation"]["embedding_dim"] == 768
        assert config["representation"].get("dedup", False) is dedup
