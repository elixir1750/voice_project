from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
import pyarrow.dataset as pa_dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from utils.tokenizer import CharacterCTCTokenizer


def normalize_audio(
    array: Any,
    source_rate: int,
    target_rate: int,
) -> torch.Tensor:
    waveform = torch.as_tensor(array, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim != 1:
        raise ValueError(f"Expected mono or channel-first audio, got {waveform.shape}")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio sample rates must be positive")
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=source_rate,
            new_freq=target_rate,
        )
    return waveform.contiguous()


def take_samples(
    iterable: Iterable[dict[str, Any]],
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("Sample limit must be non-negative or None")
    iterator = iter(iterable)
    try:
        if limit is None:
            yield from iterator
            return
        for _ in range(limit):
            try:
                yield next(iterator)
            except StopIteration:
                return
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _decode_audio(audio: Any) -> tuple[Any, int]:
    if isinstance(audio, Mapping):
        if "array" in audio and "sampling_rate" in audio:
            return audio["array"], int(audio["sampling_rate"])
        if audio.get("path"):
            array, sample_rate = sf.read(audio["path"], always_2d=False)
            if getattr(array, "ndim", 0) == 2:
                array = array.T
            return array, int(sample_rate)
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        sample_rate = getattr(samples, "sample_rate", None)
        if sample_rate is None:
            sample_rate = getattr(samples, "sampling_rate", None)
        if sample_rate is None:
            raise ValueError("Decoded audio is missing a sample rate")
        return data, int(sample_rate)
    if isinstance(audio, (str, Path)):
        array, sample_rate = sf.read(audio, always_2d=False)
        if getattr(array, "ndim", 0) == 2:
            array = array.T
        return array, int(sample_rate)
    raise ValueError(f"Unsupported audio value: {type(audio).__name__}")


class ASRCollator:
    def __init__(
        self,
        tokenizer: CharacterCTCTokenizer,
        sample_rate: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        waveforms = [
            torch.as_tensor(example["audio"], dtype=torch.float32).flatten()
            for example in examples
        ]
        texts = [self.tokenizer.normalize(str(example["text"])) for example in examples]
        encoded = [self.tokenizer.encode(text) for text in texts]
        target_tensors = [torch.tensor(ids, dtype=torch.long) for ids in encoded]
        targets = (
            torch.cat(target_tensors)
            if target_tensors
            else torch.empty(0, dtype=torch.long)
        )
        waveform_lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            dtype=torch.long,
        )
        target_lengths = torch.tensor(
            [target.numel() for target in target_tensors],
            dtype=torch.long,
        )
        return {
            "waveforms": pad_sequence(waveforms, batch_first=True),
            "waveform_lengths": waveform_lengths,
            "targets": targets,
            "target_lengths": target_lengths,
            "texts": texts,
            "audio_seconds": waveform_lengths.to(torch.float32) / self.sample_rate,
        }


class _PreparedSpeechStream(IterableDataset):
    def __init__(
        self,
        source: Iterable[dict[str, Any]],
        sample_rate: int,
        limit: int | None,
    ) -> None:
        super().__init__()
        self.source = source
        self.sample_rate = sample_rate
        self.limit = limit

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for example in take_samples(iter(self.source), self.limit):
            if "audio" not in example or "text" not in example:
                raise ValueError("ASR examples require audio and text fields")
            array, source_rate = _decode_audio(example["audio"])
            waveform = normalize_audio(array, source_rate, self.sample_rate)
            if waveform.numel() == 0:
                continue
            yield {"audio": waveform, "text": str(example["text"])}


class LibriSpeechDataModule:
    def __init__(
        self,
        config: Mapping[str, Any],
        tokenizer: CharacterCTCTokenizer,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.config = dict(config)
        self.tokenizer = tokenizer
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.sample_rate = int(self.config.get("sample_rate", 16_000))
        self.collator = ASRCollator(tokenizer, self.sample_rate)

    def _load_stream(
        self,
        split: str,
        training: bool,
        sample_limit: int | None,
    ) -> Iterable[dict[str, Any]]:
        load_arguments: dict[str, Any] = {
            "split": split,
            "streaming": bool(self.config.get("streaming", True)),
            "cache_dir": self.cache_dir,
        }
        scan_options = pa_dataset.ParquetFragmentScanOptions()
        scan_options.pre_buffer = False
        load_arguments["fragment_scan_options"] = scan_options
        if sample_limit is not None and sample_limit > 0:
            load_arguments["batch_size"] = sample_limit
        stream = load_dataset(
            self.config.get("dataset_name", "librispeech_asr"),
            self.config.get("dataset_config", "clean"),
            **load_arguments,
        )
        if sample_limit is not None and hasattr(stream, "take"):
            stream = stream.take(sample_limit)
        if training and hasattr(stream, "shuffle"):
            configured_buffer = int(self.config.get("shuffle_buffer", 256))
            buffer_size = (
                min(configured_buffer, sample_limit)
                if sample_limit is not None
                else configured_buffer
            )
            stream = stream.shuffle(
                seed=int(self.config.get("seed", 42)),
                buffer_size=buffer_size,
            )
        return stream

    def _dataloader(
        self,
        split: str,
        sample_limit: int | None,
        training: bool,
        training_config: Mapping[str, Any],
    ) -> DataLoader:
        num_workers = int(training_config.get("num_workers", 0))
        if bool(self.config.get("streaming", True)) and num_workers > 1:
            raise ValueError(
                "Streaming ASR datasets require num_workers <= 1 to avoid "
                "duplicating the limited iterable in each worker"
            )
        dataset = _PreparedSpeechStream(
            source=self._load_stream(
                split,
                training=training,
                sample_limit=sample_limit,
            ),
            sample_rate=self.sample_rate,
            limit=sample_limit,
        )
        return DataLoader(
            dataset,
            batch_size=int(training_config.get("batch_size", 1)),
            collate_fn=self.collator,
            num_workers=num_workers,
            pin_memory=bool(training_config.get("pin_memory", False)),
        )

    def train_dataloader(
        self,
        training_config: Mapping[str, Any],
    ) -> DataLoader:
        return self._dataloader(
            split=str(self.config.get("train_split", "train.100")),
            sample_limit=self.config.get("train_samples"),
            training=True,
            training_config=training_config,
        )

    def validation_dataloader(
        self,
        training_config: Mapping[str, Any],
    ) -> DataLoader:
        return self._dataloader(
            split=str(self.config.get("validation_split", "validation")),
            sample_limit=self.config.get("validation_samples"),
            training=False,
            training_config=training_config,
        )
