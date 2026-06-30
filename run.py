from __future__ import annotations

import argparse
from collections.abc import Sequence

from config import apply_overrides, load_config
from evaluate import evaluate_from_paths
from inference import transcribe_path
from train import fit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular low-resource ASR with self-supervised speech features"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an ASR experiment")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--set", action="append", default=[])
    train_parser.add_argument("--resume")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained checkpoint",
    )
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--set", action="append", default=[])

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe one audio file or all supported files in a directory",
    )
    transcribe_parser.add_argument("--checkpoint", required=True)
    transcribe_parser.add_argument("--audio", required=True)
    transcribe_parser.add_argument("--device", default="auto")
    transcribe_parser.add_argument(
        "--output",
        help="Optional JSON output path (recommended for directory input)",
    )

    experiments_parser = subparsers.add_parser(
        "experiments",
        help="Preview or execute a YAML experiment matrix",
    )
    experiments_parser.add_argument("--matrix", required=True)
    experiments_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train experiments; without this flag only preview",
    )
    experiments_parser.add_argument(
        "--name",
        dest="names",
        action="append",
        default=[],
        help="Run or preview only a named experiment; may be repeated",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        config = apply_overrides(load_config(args.config), args.set)
        fit(config, resume=args.resume)
    elif args.command == "evaluate":
        evaluate_from_paths(
            config_path=args.config,
            checkpoint=args.checkpoint,
            overrides=args.set,
        )
    elif args.command == "transcribe":
        transcribe_path(
            checkpoint=args.checkpoint,
            audio_path=args.audio,
            device=args.device,
            output_path=args.output,
        )
    elif args.command == "experiments":
        from run_experiments import run_experiment_matrix

        run_experiment_matrix(
            matrix_path=args.matrix,
            execute=args.execute,
            names=args.names,
        )
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
