from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from analysis.cluster_usage import (
    analyze_cluster_usage,
    load_assignments,
    write_cluster_usage_report,
)
from run import build_parser


def test_analyze_cluster_usage_reports_entropy_and_dead_clusters() -> None:
    result = analyze_cluster_usage([0, 0, 1, 2, 2, 2], codebook_size=4)

    expected_entropy = -sum(
        probability * math.log2(probability)
        for probability in [2 / 6, 1 / 6, 3 / 6]
    )
    assert result["total_tokens"] == 6
    assert result["counts"] == [2, 1, 3, 0]
    assert result["dead_clusters"] == 1
    assert result["dead_cluster_ratio"] == 0.25
    assert result["entropy_bits"] == expected_entropy
    assert result["perplexity"] == 2**expected_entropy
    assert result["normalized_entropy"] == expected_entropy / math.log2(4)


def test_analyze_cluster_usage_can_mark_rare_clusters_as_dead() -> None:
    result = analyze_cluster_usage(
        [0] * 95 + [1] * 5,
        codebook_size=4,
        dead_min_frequency=0.1,
    )

    assert result["dead_clusters"] == 3
    assert result["active_clusters"] == 1


def test_load_assignments_accepts_text_csv_and_json(tmp_path: Path) -> None:
    text_path = tmp_path / "tokens.txt"
    text_path.write_text("0 1 2\n2,3\n", encoding="utf-8")
    json_path = tmp_path / "tokens.json"
    json_path.write_text(json.dumps([[4, 5], [6]]), encoding="utf-8")

    assert load_assignments(text_path) == [0, 1, 2, 2, 3]
    assert load_assignments(json_path) == [4, 5, 6]


def test_write_cluster_usage_report_outputs_json_csv_and_svg(
    tmp_path: Path,
) -> None:
    first = tmp_path / "units_k4.txt"
    second = tmp_path / "units_k8.txt"
    first.write_text("0 0 1 2 2 2", encoding="utf-8")
    second.write_text("0 1 2 3 4 5 6 7", encoding="utf-8")
    output_dir = tmp_path / "report"

    rows = write_cluster_usage_report(
        assignment_paths=[first, second],
        codebook_sizes=[4, 8],
        output_dir=output_dir,
    )

    assert [row["codebook_size"] for row in rows] == [4, 8]
    assert (output_dir / "cluster_usage_summary.json").exists()
    assert (output_dir / "cluster_usage_summary.csv").exists()
    assert (output_dir / "cluster_usage_by_k.svg").exists()
    with (output_dir / "cluster_usage_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[0]["name"] == "units_k4"


def test_write_cluster_usage_report_can_join_downstream_wer(
    tmp_path: Path,
) -> None:
    assignments = tmp_path / "units_k4.txt"
    assignments.write_text("0 0 1 2 2 2", encoding="utf-8")
    metrics = tmp_path / "downstream.csv"
    metrics.write_text(
        "codebook_size,wer,cer\n4,0.42,0.12\n",
        encoding="utf-8",
    )

    rows = write_cluster_usage_report(
        assignment_paths=[assignments],
        codebook_sizes=[4],
        output_dir=tmp_path / "report",
        downstream_metrics_path=metrics,
    )

    assert rows[0]["wer"] == 0.42
    assert rows[0]["cer"] == 0.12


def test_parser_accepts_cluster_usage_command() -> None:
    args = build_parser().parse_args(
        [
            "cluster-usage",
            "--assignments",
            "units_k128.txt",
            "units_k256.txt",
            "--codebook-size",
            "128",
            "--codebook-size",
            "256",
            "--output-dir",
            "results/research_b_cluster_usage",
            "--downstream-metrics",
            "results/discrete_wer.csv",
        ]
    )

    assert args.command == "cluster-usage"
    assert args.assignments == ["units_k128.txt", "units_k256.txt"]
    assert args.codebook_sizes == [128, 256]
    assert args.downstream_metrics == "results/discrete_wer.csv"
