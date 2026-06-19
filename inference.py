from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio

from data.dataset import normalize_audio
from train import build_model_from_config
from utils.checkpoint import load_checkpoint, restore_checkpoint
from utils.device import resolve_device
from utils.metrics import compute_rtf
from utils.tokenizer import CharacterCTCTokenizer


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


@torch.no_grad()
def transcribe_audio(
    checkpoint: str | Path,
    audio_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    config = payload["config"]
    resolved_device = resolve_device(device)
    tokenizer = CharacterCTCTokenizer.from_state_dict(payload["tokenizer"])
    model = build_model_from_config(config, tokenizer).to(resolved_device)
    restore_checkpoint(payload, model=model, strict=True)
    model.eval()

    sample_rate = int(config["data"].get("sample_rate", 16_000))
    waveform = load_audio(audio_path, sample_rate)
    waveforms = waveform.unsqueeze(0).to(resolved_device)
    lengths = torch.tensor(
        [waveform.numel()],
        dtype=torch.long,
        device=resolved_device,
    )
    _synchronize(resolved_device)
    started_at = time.perf_counter()
    output = model(waveforms, lengths)
    _synchronize(resolved_device)
    inference_seconds = time.perf_counter() - started_at

    token_ids = output.logits.argmax(dim=-1)[0, : int(output.lengths[0])]
    result = {
        "text": tokenizer.decode_ctc(token_ids.detach().cpu().tolist()),
        "audio_seconds": waveform.numel() / sample_rate,
        "inference_seconds": inference_seconds,
        "rtf": compute_rtf(waveform.numel() / sample_rate, inference_seconds),
        "device": str(resolved_device),
    }
    print(json.dumps(result, ensure_ascii=False))
    return result
