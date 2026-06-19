from __future__ import annotations

from collections.abc import Sequence

import jiwer


def compute_error_rates(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal length")
    if not references:
        raise ValueError("At least one reference is required")
    return {
        "wer": float(jiwer.wer(list(references), list(hypotheses))),
        "cer": float(jiwer.cer(list(references), list(hypotheses))),
    }


def compute_rtf(audio_seconds: float, inference_seconds: float) -> float:
    if audio_seconds <= 0:
        raise ValueError("Audio duration must be positive")
    if inference_seconds < 0:
        raise ValueError("Inference duration cannot be negative")
    return inference_seconds / audio_seconds
