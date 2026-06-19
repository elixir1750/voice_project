from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import apply_overrides, load_config
from data.dataset import LibriSpeechDataModule
from train import build_model_from_config, validate
from utils.checkpoint import load_checkpoint, restore_checkpoint
from utils.device import resolve_device
from utils.tokenizer import CharacterCTCTokenizer


def evaluate_checkpoint(
    checkpoint: str | Path,
    config: Mapping[str, Any] | None = None,
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    evaluation_config = dict(config or payload["config"])
    evaluation_config = apply_overrides(evaluation_config, overrides)
    device = resolve_device(
        str(evaluation_config["runtime"].get("device", "auto"))
    )
    tokenizer = CharacterCTCTokenizer.from_state_dict(payload["tokenizer"])
    model = build_model_from_config(evaluation_config, tokenizer).to(device)
    restore_checkpoint(payload, model=model, strict=True)
    data_module = LibriSpeechDataModule(
        config=evaluation_config["data"],
        tokenizer=tokenizer,
        cache_dir=evaluation_config["runtime"].get("cache_dir"),
    )
    metrics = validate(
        model=model,
        dataloader=data_module.validation_dataloader(
            evaluation_config["training"]
        ),
        tokenizer=tokenizer,
        device=device,
    )
    result = {
        key: value
        for key, value in metrics.items()
        if key not in {"references", "hypotheses"}
    }
    result["device"] = str(device)

    output_dir = Path(evaluation_config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return result


def evaluate_from_paths(
    config_path: str | Path,
    checkpoint: str | Path,
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    return evaluate_checkpoint(
        checkpoint=checkpoint,
        config=load_config(config_path),
        overrides=overrides,
    )
