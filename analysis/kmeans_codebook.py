from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm

from data.dataset import LibriSpeechDataModule
from models.registry import build_ssl
from utils.device import resolve_device
from utils.tokenizer import CharacterCTCTokenizer


def _valid_frames(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    frames = []
    for row, length_tensor in zip(values, lengths):
        length = int(length_tensor.item())
        if length > 0:
            frames.append(row[:length].detach().cpu())
    if not frames:
        return torch.empty(0, values.shape[-1])
    return torch.cat(frames, dim=0)


def fit_kmeans_codebook(
    config: Mapping[str, Any],
    codebook_size: int,
    output_path: str | Path,
    frame_sample_limit: int = 500_000,
    batch_size: int | None = None,
    device: str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if frame_sample_limit <= 0:
        raise ValueError("frame_sample_limit must be positive")

    resolved_device = resolve_device(device)
    ssl_config = dict(config["ssl"])
    cache_dir = config.get("runtime", {}).get("cache_dir")
    if cache_dir and "cache_dir" not in ssl_config:
        ssl_config["cache_dir"] = cache_dir
    ssl = build_ssl(ssl_config).to(resolved_device)
    ssl.eval()

    tokenizer = CharacterCTCTokenizer()
    data_module = LibriSpeechDataModule(
        config=config["data"],
        tokenizer=tokenizer,
        cache_dir=config["runtime"].get("cache_dir"),
    )
    training_config = dict(config["training"])
    if batch_size is not None:
        training_config["batch_size"] = batch_size
    training_config["num_workers"] = int(training_config.get("num_workers", 0))

    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size,
        random_state=random_state,
        batch_size=min(10_000, max(codebook_size * 4, 1024)),
        n_init=3,
    )
    buffered: list[torch.Tensor] = []
    buffered_frames = 0
    frames_seen = 0
    initialized = False

    with torch.no_grad():
        for batch in tqdm(
            data_module.train_dataloader(training_config),
            desc=f"fit-kmeans-k{codebook_size}",
        ):
            waveforms = batch["waveforms"].to(resolved_device)
            waveform_lengths = batch["waveform_lengths"].to(resolved_device)
            features = ssl(waveforms, waveform_lengths)
            valid = _valid_frames(features.values, features.lengths)
            if valid.numel() == 0:
                continue
            remaining = frame_sample_limit - frames_seen
            if remaining <= 0:
                break
            valid = valid[:remaining]
            frames_seen += int(valid.shape[0])
            buffered.append(valid)
            buffered_frames += int(valid.shape[0])
            if buffered_frames >= max(codebook_size, 1024):
                chunk = torch.cat(buffered, dim=0).numpy()
                kmeans.partial_fit(chunk)
                initialized = True
                buffered.clear()
                buffered_frames = 0
            if frames_seen >= frame_sample_limit:
                break

    if buffered:
        chunk = torch.cat(buffered, dim=0).numpy()
        if not initialized and chunk.shape[0] < codebook_size:
            raise ValueError(
                f"Need at least {codebook_size} frames to initialize k-means, "
                f"got {chunk.shape[0]}"
            )
        kmeans.partial_fit(chunk)
        initialized = True
    if not initialized:
        raise RuntimeError("No frames were available to fit k-means")

    payload = {
        "model": kmeans,
        "codebook_size": codebook_size,
        "input_dim": int(ssl.output_dim),
        "ssl": dict(config["ssl"]),
        "train_split": config["data"].get("train_split"),
        "train_samples": config["data"].get("train_samples"),
        "frame_sample_limit": frame_sample_limit,
        "frames_seen": frames_seen,
        "random_state": random_state,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pickle.dumps(payload))
    return {
        key: value
        for key, value in payload.items()
        if key != "model"
    }


def fit_many_kmeans_codebooks(
    config: Mapping[str, Any],
    codebook_sizes: Sequence[int],
    output_dir: str | Path,
    frame_sample_limit: int = 500_000,
    device: str = "auto",
    random_state: int = 42,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    results = []
    layer = config["ssl"].get("layer", "unknown")
    for codebook_size in codebook_sizes:
        output_path = output_root / f"wav2vec2_layer{layer}_k{codebook_size}.pkl"
        results.append(
            fit_kmeans_codebook(
                config=config,
                codebook_size=codebook_size,
                output_path=output_path,
                frame_sample_limit=frame_sample_limit,
                device=device,
                random_state=random_state,
            )
        )
    return results
