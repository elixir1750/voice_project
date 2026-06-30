from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def _flatten_tokens(value: Any) -> list[int]:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid cluster ids")
    if isinstance(value, int):
        return [value]
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Cluster id must be an integer, got {value}")
        return [int(value)]
    if isinstance(value, str):
        return [int(value)]
    if isinstance(value, Iterable):
        tokens: list[int] = []
        for item in value:
            tokens.extend(_flatten_tokens(item))
        return tokens
    raise ValueError(f"Unsupported assignment value: {value!r}")


def load_assignments(path: str | Path) -> list[int]:
    assignment_path = Path(path)
    suffix = assignment_path.suffix.lower()
    if suffix == ".json":
        return _flatten_tokens(
            json.loads(assignment_path.read_text(encoding="utf-8"))
        )
    if suffix in {".npy", ".npz"}:
        import numpy as np

        payload = np.load(assignment_path, allow_pickle=False)
        if suffix == ".npz":
            if "assignments" in payload:
                array = payload["assignments"]
            else:
                keys = sorted(payload.files)
                if not keys:
                    raise ValueError(f"No arrays found in {assignment_path}")
                array = payload[keys[0]]
        else:
            array = payload
        return _flatten_tokens(array.tolist())

    text = assignment_path.read_text(encoding="utf-8")
    values = [item for item in re.split(r"[\s,]+", text.strip()) if item]
    return _flatten_tokens(values)


def analyze_cluster_usage(
    assignments: Sequence[int],
    codebook_size: int,
    dead_min_count: int = 0,
    dead_min_frequency: float = 0.0,
) -> dict[str, Any]:
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if dead_min_count < 0:
        raise ValueError("dead_min_count must be non-negative")
    if not 0.0 <= dead_min_frequency < 1.0:
        raise ValueError("dead_min_frequency must be in [0, 1)")
    if not assignments:
        raise ValueError("assignments must not be empty")

    counts = [0 for _ in range(codebook_size)]
    for token in assignments:
        if token < 0 or token >= codebook_size:
            raise ValueError(
                f"Cluster id {token} is outside [0, {codebook_size})"
            )
        counts[token] += 1

    total_tokens = sum(counts)
    probabilities = [count / total_tokens for count in counts]
    entropy_bits = -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0.0
    )
    max_entropy_bits = math.log2(codebook_size) if codebook_size > 1 else 0.0
    normalized_entropy = (
        entropy_bits / max_entropy_bits if max_entropy_bits > 0.0 else 1.0
    )
    perplexity = 2**entropy_bits
    dead_clusters = sum(
        1
        for count, probability in zip(counts, probabilities)
        if count <= dead_min_count
        or (
            dead_min_frequency > 0.0
            and probability < dead_min_frequency
        )
    )
    most_used_count = max(counts)
    most_used_cluster = counts.index(most_used_count)

    return {
        "codebook_size": codebook_size,
        "total_tokens": total_tokens,
        "counts": counts,
        "probabilities": probabilities,
        "entropy_bits": entropy_bits,
        "max_entropy_bits": max_entropy_bits,
        "normalized_entropy": normalized_entropy,
        "perplexity": perplexity,
        "perplexity_ratio": perplexity / codebook_size,
        "dead_clusters": dead_clusters,
        "dead_cluster_ratio": dead_clusters / codebook_size,
        "active_clusters": codebook_size - dead_clusters,
        "most_used_cluster": most_used_cluster,
        "most_used_frequency": most_used_count / total_tokens,
    }


def _write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "assignment_path",
        "codebook_size",
        "total_tokens",
        "entropy_bits",
        "normalized_entropy",
        "perplexity",
        "perplexity_ratio",
        "dead_clusters",
        "dead_cluster_ratio",
        "active_clusters",
        "most_used_cluster",
        "most_used_frequency",
        "wer",
        "cer",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_downstream_metrics(
    path: str | Path | None,
) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    metrics_path = Path(path)
    if metrics_path.suffix.lower() == ".json":
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = list(payload.values())
    else:
        with metrics_path.open(newline="", encoding="utf-8") as file:
            payload = list(csv.DictReader(file))
    if not isinstance(payload, list):
        raise ValueError("downstream metrics must be a list or CSV table")

    metrics: dict[str, dict[str, float]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each downstream metric row must be a mapping")
        keys = []
        if item.get("name"):
            keys.append(str(item["name"]))
        if item.get("experiment"):
            keys.append(str(item["experiment"]))
        if item.get("codebook_size") not in {None, ""}:
            keys.append(f"k={int(item['codebook_size'])}")
        values = {
            metric: float(item[metric])
            for metric in ("wer", "cer")
            if item.get(metric) not in {None, ""}
        }
        for key in keys:
            metrics[key] = values
    return metrics


def _find_downstream_metrics(
    metrics: dict[str, dict[str, float]],
    name: str,
    codebook_size: int,
) -> dict[str, float]:
    return metrics.get(name, metrics.get(f"k={codebook_size}", {}))


def _scale(
    value: float,
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float,
) -> float:
    if old_max == old_min:
        return (new_min + new_max) / 2
    return new_min + (value - old_min) * (new_max - new_min) / (
        old_max - old_min
    )


def _polyline(points: Sequence[tuple[float, float]], color: str) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />'
        for x, y in points
    )
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{color}" '
        f'stroke-width="2" />\n{circles}'
    )


def write_cluster_usage_svg(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    sorted_rows = sorted(rows, key=lambda row: row["codebook_size"])
    width, height = 760, 420
    left, right, top, bottom = 70, 40, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row["codebook_size"]) for row in sorted_rows]
    entropy_values = [float(row["normalized_entropy"]) for row in sorted_rows]
    dead_values = [float(row["dead_cluster_ratio"]) for row in sorted_rows]
    x_min, x_max = min(x_values), max(x_values)
    entropy_points = [
        (
            _scale(x, x_min, x_max, left, left + plot_width),
            _scale(y, 0.0, 1.0, top + plot_height, top),
        )
        for x, y in zip(x_values, entropy_values)
    ]
    dead_points = [
        (
            _scale(x, x_min, x_max, left, left + plot_width),
            _scale(y, 0.0, 1.0, top + plot_height, top),
        )
        for x, y in zip(x_values, dead_values)
    ]
    x_labels = "\n".join(
        f'<text x="{_scale(x, x_min, x_max, left, left + plot_width):.1f}" '
        f'y="{height - 35}" text-anchor="middle" font-size="12">{int(x)}</text>'
        for x in x_values
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="24" text-anchor="middle" font-size="18">Cluster usage by codebook size</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" />
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" />
<text x="20" y="{top + 8}" font-size="12">1.0</text>
<text x="20" y="{top + plot_height}" font-size="12">0.0</text>
{x_labels}
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="13">Codebook size K</text>
<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-size="13">Ratio</text>
{_polyline(entropy_points, "#1f77b4")}
{_polyline(dead_points, "#d62728")}
<rect x="{width - 235}" y="42" width="180" height="54" fill="white" stroke="#ccc" />
<line x1="{width - 220}" y1="60" x2="{width - 190}" y2="60" stroke="#1f77b4" stroke-width="2" />
<text x="{width - 180}" y="64" font-size="12">normalized entropy</text>
<line x1="{width - 220}" y1="82" x2="{width - 190}" y2="82" stroke="#d62728" stroke-width="2" />
<text x="{width - 180}" y="86" font-size="12">dead cluster ratio</text>
</svg>
"""
    Path(path).write_text(svg, encoding="utf-8")


def _normalize_codebook_sizes(
    assignment_paths: Sequence[Path],
    codebook_sizes: Sequence[int],
) -> list[int]:
    if len(codebook_sizes) == 1:
        return [codebook_sizes[0] for _ in assignment_paths]
    if len(codebook_sizes) != len(assignment_paths):
        raise ValueError(
            "Provide either one codebook size or one per assignment file"
        )
    return list(codebook_sizes)


def write_cluster_usage_report(
    assignment_paths: Sequence[str | Path],
    codebook_sizes: Sequence[int],
    output_dir: str | Path,
    dead_min_count: int = 0,
    dead_min_frequency: float = 0.0,
    downstream_metrics_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    paths = [Path(path) for path in assignment_paths]
    if not paths:
        raise ValueError("assignment_paths must not be empty")
    normalized_sizes = _normalize_codebook_sizes(paths, codebook_sizes)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    downstream_metrics = _load_downstream_metrics(downstream_metrics_path)

    rows: list[dict[str, Any]] = []
    for assignment_path, codebook_size in zip(paths, normalized_sizes):
        metrics = analyze_cluster_usage(
            load_assignments(assignment_path),
            codebook_size=codebook_size,
            dead_min_count=dead_min_count,
            dead_min_frequency=dead_min_frequency,
        )
        rows.append(
            {
                "name": assignment_path.stem,
                "assignment_path": str(assignment_path),
                **metrics,
                **_find_downstream_metrics(
                    downstream_metrics,
                    name=assignment_path.stem,
                    codebook_size=codebook_size,
                ),
            }
        )

    (report_dir / "cluster_usage_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(rows, report_dir / "cluster_usage_summary.csv")
    write_cluster_usage_svg(rows, report_dir / "cluster_usage_by_k.svg")
    return rows
