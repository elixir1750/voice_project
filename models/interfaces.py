from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class SpeechFeatures:
    values: torch.Tensor
    lengths: torch.Tensor
    feature_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    lengths: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class SSLExtractor(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> SpeechFeatures:
        raise NotImplementedError


class RepresentationAdapter(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: SpeechFeatures) -> SpeechFeatures:
        raise NotImplementedError


class CTCDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DecoderOutput:
        raise NotImplementedError
