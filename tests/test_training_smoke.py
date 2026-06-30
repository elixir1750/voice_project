from __future__ import annotations

import math

import torch
from torch import nn

from models.interfaces import DecoderOutput
from train import (
    ASRModel,
    count_model_parameters,
    ctc_compute_device,
    should_save_checkpoints,
    should_stop_early,
    train_one_epoch,
    validate,
)
from utils.tokenizer import CharacterCTCTokenizer


class _TinyASRModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, vocab_size)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> DecoderOutput:
        del waveform_lengths
        frames = waveforms[:, :4].unsqueeze(-1)
        lengths = torch.full(
            (waveforms.shape[0],),
            fill_value=frames.shape[1],
            dtype=torch.long,
            device=waveforms.device,
        )
        return DecoderOutput(logits=self.projection(frames), lengths=lengths)


def _batch(tokenizer: CharacterCTCTokenizer) -> dict:
    targets = [
        tokenizer.token_to_id["h"],
        tokenizer.token_to_id["i"],
        tokenizer.token_to_id["a"],
    ]
    return {
        "waveforms": torch.randn(2, 8),
        "waveform_lengths": torch.tensor([8, 8]),
        "targets": torch.tensor(targets),
        "target_lengths": torch.tensor([2, 1]),
        "texts": ["hi", "a"],
        "audio_seconds": torch.tensor([0.5, 0.5]),
    }


def test_one_fake_training_epoch_returns_finite_loss() -> None:
    tokenizer = CharacterCTCTokenizer()
    model = _TinyASRModel(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    result = train_one_epoch(
        model=model,
        dataloader=[_batch(tokenizer)],
        optimizer=optimizer,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        grad_clip=1.0,
        amp=False,
    )

    assert math.isfinite(result["loss"])
    assert result["batches"] == 1
    assert result["optimizer_steps"] == 1


def test_gradient_accumulation_steps_final_partial_group() -> None:
    tokenizer = CharacterCTCTokenizer()
    model = _TinyASRModel(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    original_step = optimizer.step
    step_calls = 0

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step

    result = train_one_epoch(
        model=model,
        dataloader=[_batch(tokenizer) for _ in range(3)],
        optimizer=optimizer,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        grad_clip=1.0,
        amp=False,
        gradient_accumulation_steps=2,
    )

    assert step_calls == 2
    assert result["batches"] == 3
    assert result["optimizer_steps"] == 2


def test_fake_validation_returns_metrics_and_hypotheses() -> None:
    tokenizer = CharacterCTCTokenizer()
    model = _TinyASRModel(len(tokenizer))

    result = validate(
        model=model,
        dataloader=[_batch(tokenizer)],
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    assert math.isfinite(result["loss"])
    assert math.isfinite(result["wer"])
    assert math.isfinite(result["cer"])
    assert math.isfinite(result["rtf"])
    assert result["num_samples"] == 2
    assert len(result["hypotheses"]) == 2


def test_ctc_loss_falls_back_to_cpu_for_mps() -> None:
    assert ctc_compute_device(torch.device("mps")) == torch.device("cpu")
    assert ctc_compute_device(torch.device("cuda")) == torch.device("cuda")
    assert ctc_compute_device(torch.device("cpu")) == torch.device("cpu")


def test_early_stopping_can_be_disabled_or_triggered() -> None:
    assert not should_stop_early(epochs_without_improvement=10, patience=0)
    assert not should_stop_early(epochs_without_improvement=2, patience=3)
    assert should_stop_early(epochs_without_improvement=3, patience=3)


def test_checkpoint_saving_defaults_on_and_can_be_disabled() -> None:
    assert should_save_checkpoints({}) is True
    assert should_save_checkpoints({"save_checkpoints": True}) is True
    assert should_save_checkpoints({"save_checkpoints": False}) is False


def test_count_model_parameters_separates_decoder_parameters() -> None:
    ssl = nn.Linear(3, 5)
    for parameter in ssl.parameters():
        parameter.requires_grad = False
    representation = nn.Identity()
    decoder = nn.Linear(5, 7)
    model = ASRModel(ssl=ssl, representation=representation, decoder=decoder)

    result = count_model_parameters(model)

    assert result == {
        "total": 62,
        "trainable": 42,
        "decoder": 42,
        "decoder_trainable": 42,
    }
