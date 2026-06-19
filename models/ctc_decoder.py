from __future__ import annotations

import torch
from torch import nn

from models.interfaces import CTCDecoder, DecoderOutput
from models.registry import register_decoder


@register_decoder("linear")
class LinearCTCDecoder(CTCDecoder):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, vocab_size),
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DecoderOutput:
        return DecoderOutput(logits=self.network(features), lengths=lengths)


@register_decoder("mlp")
class MLPCTCDecoder(CTCDecoder):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DecoderOutput:
        return DecoderOutput(logits=self.network(features), lengths=lengths)
