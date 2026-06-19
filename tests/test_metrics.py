from __future__ import annotations

import pytest

from utils.metrics import compute_error_rates, compute_rtf


def test_error_rates_detect_transcription_errors() -> None:
    result = compute_error_rates(["hello world"], ["hello word"])

    assert result["wer"] > 0
    assert result["cer"] > 0


def test_error_rates_are_zero_for_identical_text() -> None:
    result = compute_error_rates(["hello world"], ["hello world"])

    assert result == {"wer": 0.0, "cer": 0.0}


def test_rtf_is_inference_time_divided_by_audio_time() -> None:
    assert compute_rtf(audio_seconds=2.0, inference_seconds=1.0) == 0.5


def test_rtf_rejects_non_positive_audio_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_rtf(audio_seconds=0.0, inference_seconds=1.0)
