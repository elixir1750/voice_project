from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from data.dataset import (
    ASRCollator,
    LibriSpeechDataModule,
    _decode_audio,
    normalize_audio,
    take_samples,
)
from utils.tokenizer import CharacterCTCTokenizer


def test_collator_pads_audio_and_flattens_targets() -> None:
    tokenizer = CharacterCTCTokenizer()
    collator = ASRCollator(tokenizer=tokenizer, sample_rate=16_000)

    batch = collator(
        [
            {"audio": torch.ones(4), "text": "hi"},
            {"audio": torch.ones(2), "text": "a"},
        ]
    )

    assert batch["waveforms"].shape == (2, 4)
    assert batch["waveform_lengths"].tolist() == [4, 2]
    assert batch["target_lengths"].tolist() == [2, 1]
    assert batch["targets"].ndim == 1
    assert batch["targets"].numel() == 3
    assert batch["texts"] == ["hi", "a"]


def test_normalize_audio_converts_stereo_to_mono() -> None:
    stereo = torch.stack([torch.ones(8), torch.zeros(8)])

    mono = normalize_audio(stereo, source_rate=16_000, target_rate=16_000)

    assert mono.shape == (8,)
    assert torch.allclose(mono, torch.full((8,), 0.5))


def test_soundfile_stereo_preserves_time_axis(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    channel_last = np.stack(
        [np.ones(100, dtype=np.float32), np.zeros(100, dtype=np.float32)],
        axis=1,
    )
    sf.write(path, channel_last, 16_000)

    decoded, sample_rate = _decode_audio(path)
    mono = normalize_audio(decoded, sample_rate, 16_000)

    assert mono.shape == (100,)
    assert torch.allclose(mono, torch.full((100,), 0.5), atol=1e-4)


def test_normalize_audio_resamples() -> None:
    audio = torch.ones(8_000)

    resampled = normalize_audio(audio, source_rate=8_000, target_rate=16_000)

    assert 15_990 <= resampled.numel() <= 16_010
    assert resampled.dtype == torch.float32


def test_take_samples_applies_limit_without_overreading() -> None:
    consumed: list[int] = []

    def source():
        for item in range(10):
            consumed.append(item)
            yield {"id": item}

    samples = list(take_samples(source(), 3))

    assert [sample["id"] for sample in samples] == [0, 1, 2]
    assert consumed == [0, 1, 2]


def test_take_samples_accepts_unlimited_stream() -> None:
    source = [{"id": 1}, {"id": 2}]

    assert list(take_samples(source, None)) == source


def test_take_samples_closes_limited_source_iterator() -> None:
    class ClosableIterator:
        def __init__(self) -> None:
            self.current = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            value = self.current
            self.current += 1
            return {"id": value}

        def close(self) -> None:
            self.closed = True

    source = ClosableIterator()

    assert len(list(take_samples(source, 2))) == 2
    assert source.closed is True


def test_data_module_limits_stream_before_shuffle(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int]] = []
    load_arguments: dict = {}

    class FakeStream:
        def take(self, limit: int):
            calls.append(("take", limit))
            return self

        def shuffle(self, seed: int, buffer_size: int):
            del seed
            calls.append(("shuffle", buffer_size))
            return self

    def fake_load_dataset(*args, **kwargs):
        del args
        load_arguments.update(kwargs)
        return FakeStream()

    monkeypatch.setattr("data.dataset.load_dataset", fake_load_dataset)
    module = LibriSpeechDataModule(
        config={
            "dataset_name": "librispeech_asr",
            "dataset_config": "clean",
            "sample_rate": 16_000,
            "shuffle_buffer": 256,
        },
        tokenizer=CharacterCTCTokenizer(),
    )

    module._load_stream("train.100", training=True, sample_limit=64)

    assert load_arguments["batch_size"] == 64
    assert load_arguments["fragment_scan_options"].pre_buffer is False
    assert calls == [("take", 64), ("shuffle", 64)]


def test_streaming_data_rejects_multiple_workers() -> None:
    module = LibriSpeechDataModule(
        config={"streaming": True, "sample_rate": 16_000},
        tokenizer=CharacterCTCTokenizer(),
    )

    with pytest.raises(ValueError, match="num_workers"):
        module._dataloader(
            split="validation",
            sample_limit=2,
            training=False,
            training_config={"batch_size": 1, "num_workers": 2},
        )
