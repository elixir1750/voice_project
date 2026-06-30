from __future__ import annotations

import pytest

from utils.metrics import compute_error_rates, compute_rtf


def test_error_rates_detect_transcription_errors() -> None:
    result = compute_error_rates(["hello world"], ["hello word"])

    assert result["wer"] > 0
    assert result["cer"] > 0


def test_error_rates_are_zero_for_identical_text() -> None:
    result = compute_error_rates(["hello world"], ["hello world"])

    assert result == {
        "wer": 0.0,
        "cer": 0.0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "hits": 2,
        "reference_words": 2,
    }


def test_error_rates_report_word_error_counts() -> None:
    result = compute_error_rates(
        ["one two three", "four five"],
        ["one too three extra", "four"],
    )

    assert result["substitutions"] == 1
    assert result["deletions"] == 1
    assert result["insertions"] == 1
    assert result["hits"] == 3
    assert result["reference_words"] == 5
    assert result["wer"] == pytest.approx(3 / 5)


def test_rtf_is_inference_time_divided_by_audio_time() -> None:
    assert compute_rtf(audio_seconds=2.0, inference_seconds=1.0) == 0.5


def test_rtf_rejects_non_positive_audio_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_rtf(audio_seconds=0.0, inference_seconds=1.0)
