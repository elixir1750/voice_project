from __future__ import annotations

from models.interfaces import RepresentationAdapter, SpeechFeatures
from models.registry import register_representation


@register_representation("continuous")
class ContinuousRepresentation(RepresentationAdapter):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._output_dim = input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, features: SpeechFeatures) -> SpeechFeatures:
        return features
