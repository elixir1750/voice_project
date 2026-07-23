from __future__ import annotations

import math

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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_length: int = 4096) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / model_dim)
        )
        encoding = torch.zeros(max_length, model_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {values.shape[1]} exceeds positional "
                f"encoding limit {self.encoding.shape[1]}"
            )
        return values + self.encoding[:, : values.shape[1]].to(
            dtype=values.dtype
        )


@register_decoder("transformer")
class TransformerCTCDecoder(CTCDecoder):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        if min(
            input_dim,
            vocab_size,
            model_dim,
            num_heads,
            num_layers,
            feedforward_dim,
        ) <= 0:
            raise ValueError("Transformer dimensions and layer counts must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = SinusoidalPositionalEncoding(model_dim, max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(model_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.last_padding_mask = torch.empty(0, dtype=torch.bool)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DecoderOutput:
        if features.ndim != 3:
            raise ValueError("Transformer features must have shape [batch, time, dim]")
        time_steps = features.shape[1]
        positions = torch.arange(time_steps, device=features.device).unsqueeze(0)
        padding_mask = positions >= lengths.to(features.device).unsqueeze(1)
        self.last_padding_mask = padding_mask.detach()
        hidden = self.input_projection(features)
        hidden = self.dropout(self.position(hidden))
        hidden = self.encoder(
            hidden,
            src_key_padding_mask=padding_mask,
        )
        return DecoderOutput(
            logits=self.output_projection(hidden),
            lengths=lengths,
        )
