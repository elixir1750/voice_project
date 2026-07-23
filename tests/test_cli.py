from __future__ import annotations

import pytest

from run import build_parser


def test_parser_accepts_train_command() -> None:
    args = build_parser().parse_args(
        ["train", "--config", "configs/quick_test.yaml"]
    )

    assert args.command == "train"
    assert args.config == "configs/quick_test.yaml"


def test_parser_accepts_evaluate_command() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--config",
            "configs/quick_test.yaml",
            "--checkpoint",
            "outputs/quick_test/best.pt",
        ]
    )

    assert args.command == "evaluate"
    assert args.checkpoint.endswith("best.pt")


def test_parser_accepts_transcribe_command() -> None:
    args = build_parser().parse_args(
        [
            "transcribe",
            "--checkpoint",
            "outputs/quick_test/best.pt",
            "--audio",
            "sample.flac",
            "--output",
            "outputs/transcriptions.json",
        ]
    )

    assert args.command == "transcribe"
    assert args.device == "auto"
    assert args.output == "outputs/transcriptions.json"


def test_parser_accepts_safe_experiments_dry_run() -> None:
    args = build_parser().parse_args(
        ["experiments", "--matrix", "configs/experiments.yaml"]
    )

    assert args.command == "experiments"
    assert args.execute is False
    assert args.names == []


def test_parser_accepts_fit_kmeans_command() -> None:
    args = build_parser().parse_args(
        [
            "fit-kmeans",
            "--config",
            "configs/baseline.yaml",
            "--set",
            "ssl.layer=9",
            "--codebook-size",
            "100",
            "--codebook-size",
            "500",
            "--output-dir",
            "artifacts/kmeans",
            "--frame-sample-limit",
            "300000",
        ]
    )

    assert args.command == "fit-kmeans"
    assert args.codebook_sizes == [100, 500]
    assert args.frame_sample_limit == 300000
