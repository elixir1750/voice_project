from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device is unavailable")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(normalized)


def supports_amp(device: torch.device) -> bool:
    return device.type == "cuda"
