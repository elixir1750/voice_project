from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from transformers import AutoModel

from models.interfaces import SSLExtractor, SpeechFeatures
from models.registry import register_ssl


def conv_output_lengths(
    input_lengths: torch.Tensor,
    kernels: Sequence[int],
    strides: Sequence[int],
) -> torch.Tensor:
    if len(kernels) != len(strides):
        raise ValueError("Convolution kernels and strides must have equal length")
    output_lengths = input_lengths.to(dtype=torch.long)
    for kernel, stride in zip(kernels, strides):
        output_lengths = torch.div(
            output_lengths - kernel,
            stride,
            rounding_mode="floor",
        ) + 1
    return output_lengths.clamp_min(0)


@register_ssl("wav2vec2")
class Wav2Vec2Extractor(SSLExtractor):
    def __init__(
        self,
        model_name: str,
        frozen: bool = True,
        layer: int = -1,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.frozen = frozen
        self.layer = layer
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.config.output_hidden_states = True

        if frozen:
            self.model.requires_grad_(False)
            self.model.eval()

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def train(self, mode: bool = True) -> "Wav2Vec2Extractor":
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> SpeechFeatures:
        sample_positions = torch.arange(
            waveforms.shape[1],
            device=waveforms.device,
        ).unsqueeze(0)
        attention_mask = sample_positions < waveform_lengths.unsqueeze(1)

        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            outputs = self.model(
                input_values=waveforms,
                attention_mask=attention_mask.to(dtype=torch.long),
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("SSL model did not return hidden states")
        layer_index = self.layer if self.layer >= 0 else len(hidden_states) - 1
        if not 0 <= layer_index < len(hidden_states):
            raise ValueError(
                f"Requested layer {self.layer} outside available range "
                f"0..{len(hidden_states) - 1}"
            )
        values = hidden_states[layer_index]
        lengths = conv_output_lengths(
            waveform_lengths,
            kernels=self.model.config.conv_kernel,
            strides=self.model.config.conv_stride,
        ).clamp_max(values.shape[1])
        return SpeechFeatures(
            values=values,
            lengths=lengths,
            feature_dim=self.output_dim,
            metadata={
                "model_name": self.model_name,
                "layer": layer_index,
                "frozen": self.frozen,
            },
        )
