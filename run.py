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

    cluster_parser = subparsers.add_parser(
        "cluster-usage",
        help="Analyze unlabeled k-means unit usage for codebook utilization",
    )
    cluster_parser.add_argument(
        "--assignments",
        nargs="+",
        required=True,
        help="One or more token assignment files (.txt/.csv/.json/.npy/.npz)",
    )
    cluster_parser.add_argument(
        "--codebook-size",
        dest="codebook_sizes",
        type=int,
        action="append",
        required=True,
        help=(
            "Codebook size K. Provide once for all files or repeat once per "
            "assignment file"
        ),
    )
    cluster_parser.add_argument("--output-dir", required=True)
    cluster_parser.add_argument(
        "--dead-min-count",
        type=int,
        default=0,
        help="Treat clusters with count <= this value as dead",
    )
    cluster_parser.add_argument(
        "--dead-min-frequency",
        type=float,
        default=0.0,
        help="Also treat clusters with usage frequency below this value as dead",
    )
    cluster_parser.add_argument(
        "--downstream-metrics",
        help=(
            "Optional CSV/JSON with name or codebook_size plus wer/cer columns "
            "to join intrinsic unit metrics with downstream ASR results"
        ),
    )

    kmeans_parser = subparsers.add_parser(
        "fit-kmeans",
        help="Fit k-means codebooks on training-split SSL hidden states",
    )
    kmeans_parser.add_argument("--config", required=True)
    kmeans_parser.add_argument("--set", action="append", default=[])
    kmeans_parser.add_argument(
        "--codebook-size",
        dest="codebook_sizes",
        type=int,
        action="append",
        required=True,
        help="Codebook size K; repeat for multiple codebooks",
    )
    kmeans_parser.add_argument(
        "--output-dir",
        default="artifacts/kmeans",
        help="Directory for fitted .pkl codebooks",
    )
    kmeans_parser.add_argument(
        "--frame-sample-limit",
        type=int,
        default=500_000,
        help="Maximum number of training frames used per codebook",
    )
    kmeans_parser.add_argument("--device", default="auto")

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
    elif args.command == "cluster-usage":
        from analysis.cluster_usage import write_cluster_usage_report

        write_cluster_usage_report(
            assignment_paths=args.assignments,
            codebook_sizes=args.codebook_sizes,
            output_dir=args.output_dir,
            dead_min_count=args.dead_min_count,
            dead_min_frequency=args.dead_min_frequency,
            downstream_metrics_path=args.downstream_metrics,
        )
    elif args.command == "fit-kmeans":
        from analysis.kmeans_codebook import fit_many_kmeans_codebooks

        config = apply_overrides(load_config(args.config), args.set)
        results = fit_many_kmeans_codebooks(
            config=config,
            codebook_sizes=args.codebook_sizes,
            output_dir=args.output_dir,
            frame_sample_limit=args.frame_sample_limit,
            device=args.device,
        )
        import json

        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
