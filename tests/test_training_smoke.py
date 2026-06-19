from __future__ import annotations

import math

import torch
from torch import nn

from models.interfaces import DecoderOutput
from train import ctc_compute_device, train_one_epoch, validate
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
