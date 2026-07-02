from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

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
        features.metadata.setdefault("feature_dim", features.feature_dim)
        return features


def _load_kmeans_payload(path: str | Path) -> tuple[object, dict]:
    payload = pickle.loads(Path(path).read_bytes())
    if isinstance(payload, dict) and "model" in payload:
        metadata = {key: value for key, value in payload.items() if key != "model"}
        return payload["model"], metadata
    return payload, {}


@register_representation("kmeans")
class KMeansDiscreteRepresentation(RepresentationAdapter):
    def __init__(
        self,
        input_dim: int,
        codebook_path: str | Path,
        codebook_size: int,
        embedding_dim: int | None = None,
        dedup: bool = False,
    ) -> None:
        super().__init__()
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        self.kmeans, metadata = _load_kmeans_payload(codebook_path)
        stored_size = metadata.get("codebook_size")
        if stored_size is not None and int(stored_size) != codebook_size:
            raise ValueError(
                f"Configured codebook_size={codebook_size} does not match "
                f"stored codebook_size={stored_size}"
            )
        stored_dim = metadata.get("input_dim")
        if stored_dim is not None and int(stored_dim) != input_dim:
            raise ValueError(
                f"Configured input_dim={input_dim} does not match "
                f"stored input_dim={stored_dim}"
            )
        self.codebook_path = str(codebook_path)
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim or input_dim)
        self.dedup = bool(dedup)
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)

    @property
    def output_dim(self) -> int:
        return self.embedding_dim

    def _predict_tokens(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape[:2]
        flat = values.detach().reshape(-1, values.shape[-1]).cpu().numpy()
        predicted = self.kmeans.predict(flat)
        return torch.as_tensor(
            predicted,
            dtype=torch.long,
            device=values.device,
        ).reshape(shape)

    @staticmethod
    def _deduplicate(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.numel() <= 1:
            return tokens
        keep = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        keep[1:] = tokens[1:] != tokens[:-1]
        return tokens[keep]

    def forward(self, features: SpeechFeatures) -> SpeechFeatures:
        tokens = self._predict_tokens(features.values)
        token_sequences: list[torch.Tensor] = []
        for row, length_tensor in zip(tokens, features.lengths):
            length = int(length_tensor.item())
            valid_tokens = row[:length]
            if self.dedup:
                valid_tokens = self._deduplicate(valid_tokens)
            if valid_tokens.numel() == 0:
                valid_tokens = row[:1]
            token_sequences.append(valid_tokens)

        token_lengths = torch.tensor(
            [sequence.numel() for sequence in token_sequences],
            dtype=torch.long,
            device=features.values.device,
        )
        padded_tokens = pad_sequence(
            token_sequences,
            batch_first=True,
            padding_value=0,
        )
        embedded = self.embedding(padded_tokens)
        return SpeechFeatures(
            values=embedded,
            lengths=token_lengths,
            feature_dim=self.output_dim,
            metadata={
                **features.metadata,
                "representation": "kmeans",
                "codebook_size": self.codebook_size,
                "codebook_path": self.codebook_path,
                "dedup": self.dedup,
                "token_ids": padded_tokens.detach(),
                "token_lengths": token_lengths.detach(),
                "feature_dim": self.output_dim,
            },
        )
