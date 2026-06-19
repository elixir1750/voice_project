from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from utils.tokenizer import CharacterCTCTokenizer


REQUIRED_CHECKPOINT_KEYS = {
    "model_state",
    "optimizer_state",
    "config",
    "tokenizer",
    "training_state",
}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: Mapping[str, Any],
    tokenizer: CharacterCTCTokenizer,
    training_state: Mapping[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": deepcopy(dict(config)),
        "tokenizer": tokenizer.state_dict(),
        "training_state": deepcopy(dict(training_state)),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {', '.join(missing)}")
    return payload


def restore_checkpoint(
    payload: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> None:
    model.load_state_dict(payload["model_state"], strict=strict)
    optimizer_state = payload.get("optimizer_state")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
