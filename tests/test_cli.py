from __future__ import annotations

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
        ]
    )

    assert args.command == "transcribe"
    assert args.device == "auto"


def test_parser_accepts_experiments_command() -> None:
    args = build_parser().parse_args(
        ["experiments", "--config", "configs/ablations.yaml"]
    )

    assert args.command == "experiments"
    assert args.execute is False
