from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio

from data.dataset import normalize_audio
from train import build_model_from_config
from utils.checkpoint import load_checkpoint, restore_checkpoint
from utils.device import resolve_device
from utils.metrics import compute_rtf
from utils.tokenizer import CharacterCTCTokenizer


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    waveform, source_rate = torchaudio.load(audio_path)
    return normalize_audio(waveform, int(source_rate), sample_rate)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def discover_audio_files(directory: str | Path) -> list[Path]:
    audio_dir = Path(directory)
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory not found: {audio_dir}")
    files = sorted(
        (
            path
            for path in audio_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not files:
        raise ValueError(f"No supported audio files found in: {audio_dir}")
    return files


class ASRTranscriber:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
    ) -> None:
        payload = load_checkpoint(checkpoint, map_location="cpu")
        self.config = payload["config"]
        self.device = resolve_device(device)
        self.tokenizer = CharacterCTCTokenizer.from_state_dict(
            payload["tokenizer"]
        )
        self.model = build_model_from_config(
            self.config,
            self.tokenizer,
        ).to(self.device)
        restore_checkpoint(payload, model=self.model, strict=True)
        self.model.eval()
        self.sample_rate = int(
            self.config["data"].get("sample_rate", 16_000)
        )

    @torch.no_grad()
    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        waveform = load_audio(audio_path, self.sample_rate)
        waveforms = waveform.unsqueeze(0).to(self.device)
        lengths = torch.tensor(
            [waveform.numel()],
            dtype=torch.long,
            device=self.device,
        )
        _synchronize(self.device)
        started_at = time.perf_counter()
        output = self.model(waveforms, lengths)
        _synchronize(self.device)
        inference_seconds = time.perf_counter() - started_at

        token_ids = output.logits.argmax(dim=-1)[
            0,
            : int(output.lengths[0]),
        ]
        audio_seconds = waveform.numel() / self.sample_rate
        return {
            "file": str(Path(audio_path)),
            "text": self.tokenizer.decode_ctc(
                token_ids.detach().cpu().tolist()
            ),
            "audio_seconds": audio_seconds,
            "inference_seconds": inference_seconds,
            "rtf": compute_rtf(audio_seconds, inference_seconds),
            "device": str(self.device),
        }

    def transcribe_many(
        self,
        audio_paths: Sequence[str | Path],
    ) -> list[dict[str, Any]]:
        return [self.transcribe(path) for path in audio_paths]


def _write_results(
    results: dict[str, Any] | list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def transcribe_path(
    checkpoint: str | Path,
    audio_path: str | Path,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    source = Path(audio_path)
    transcriber = ASRTranscriber(checkpoint=checkpoint, device=device)
    if source.is_dir():
        result: dict[str, Any] | list[dict[str, Any]] = (
            transcriber.transcribe_many(discover_audio_files(source))
        )
    elif source.is_file():
        result = transcriber.transcribe(source)
    else:
        raise FileNotFoundError(f"Audio path not found: {source}")
    if output_path is not None:
        _write_results(result, output_path)
    print(json.dumps(result, ensure_ascii=False))
    return result


def transcribe_audio(
    checkpoint: str | Path,
    audio_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    result = transcribe_path(
        checkpoint=checkpoint,
        audio_path=audio_path,
        device=device,
    )
    if not isinstance(result, dict):
        raise TypeError("transcribe_audio requires a single audio file")
    return result
