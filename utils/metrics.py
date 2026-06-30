from __future__ import annotations

from collections.abc import Sequence

import jiwer


def compute_error_rates(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> dict[str, float | int]:
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal length")
    if not references:
        raise ValueError("At least one reference is required")
    word_output = jiwer.process_words(list(references), list(hypotheses))
    return {
        "wer": float(word_output.wer),
        "cer": float(jiwer.cer(list(references), list(hypotheses))),
        "substitutions": int(word_output.substitutions),
        "deletions": int(word_output.deletions),
        "insertions": int(word_output.insertions),
        "hits": int(word_output.hits),
        "reference_words": int(
            word_output.hits
            + word_output.substitutions
            + word_output.deletions
        ),
    }


def compute_rtf(audio_seconds: float, inference_seconds: float) -> float:
    if audio_seconds <= 0:
        raise ValueError("Audio duration must be positive")
    if inference_seconds < 0:
        raise ValueError("Inference duration cannot be negative")
    return inference_seconds / audio_seconds
