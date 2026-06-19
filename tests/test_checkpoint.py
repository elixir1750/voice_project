from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from utils.checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from utils.tokenizer import CharacterCTCTokenizer


def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    tokenizer = CharacterCTCTokenizer()
    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config={"experiment": {"name": "test"}},
        tokenizer=tokenizer,
        training_state={"epoch": 2, "global_step": 7, "best_wer": 0.5},
    )
    original_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    restore_checkpoint(payload, model=model, optimizer=optimizer)

    assert torch.equal(model.weight, original_weight)
    assert payload["config"]["experiment"]["name"] == "test"
    assert payload["tokenizer"]["tokens"] == tokenizer.tokens
    assert payload["training_state"]["global_step"] == 7


def test_checkpoint_loader_uses_restricted_weights_only_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.touch()
    arguments = {}

    def fake_load(path, **kwargs):
        arguments.update(kwargs)
        return {
            "model_state": {},
            "optimizer_state": None,
            "config": {},
            "tokenizer": {},
            "training_state": {},
        }

    monkeypatch.setattr(torch, "load", fake_load)

    load_checkpoint(checkpoint_path)

    assert arguments["weights_only"] is True
