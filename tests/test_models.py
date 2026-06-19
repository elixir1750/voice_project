from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.ctc_decoder import LinearCTCDecoder, MLPCTCDecoder
from models.interfaces import SpeechFeatures
from models.registry import build_decoder, build_representation, build_ssl
from models.ssl_extractor import (
    Wav2Vec2Extractor,
    conv_output_lengths,
    normalize_waveforms,
)


def test_linear_decoder_preserves_time_lengths() -> None:
    decoder = LinearCTCDecoder(input_dim=8, vocab_size=30, dropout=0.0)

    output = decoder(torch.randn(2, 5, 8), torch.tensor([5, 3]))

    assert output.logits.shape == (2, 5, 30)
    assert output.lengths.tolist() == [5, 3]


def test_mlp_decoder_preserves_time_lengths() -> None:
    decoder = MLPCTCDecoder(
        input_dim=8,
        vocab_size=30,
        hidden_dim=16,
        dropout=0.0,
    )

    output = decoder(torch.randn(2, 5, 8), torch.tensor([5, 4]))

    assert output.logits.shape == (2, 5, 30)
    assert output.lengths.tolist() == [5, 4]


def test_registry_builds_decoder_and_continuous_representation() -> None:
    decoder = build_decoder({"type": "linear", "dropout": 0.0}, 8, 30)
    representation = build_representation({"type": "continuous"}, 8)
    features = SpeechFeatures(
        values=torch.randn(2, 5, 8),
        lengths=torch.tensor([5, 3]),
        feature_dim=8,
    )

    represented = representation(features)

    assert isinstance(decoder, LinearCTCDecoder)
    assert represented is features
    assert representation.output_dim == 8


def test_registry_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="Unknown decoder"):
        build_decoder({"type": "mystery"}, 8, 30)


def test_wav2vec_feature_length_formula() -> None:
    lengths = conv_output_lengths(
        torch.tensor([16_000, 8_000]),
        kernels=[10, 3, 3, 3, 3, 2, 2],
        strides=[5, 2, 2, 2, 2, 2, 2],
    )

    assert lengths.tolist() == [49, 24]


class _FakeSSLModel(nn.Module):
    def __init__(self, feat_extract_norm: str = "group") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.forward_kwargs = {}
        self.config = SimpleNamespace(
            hidden_size=4,
            conv_kernel=[4, 2],
            conv_stride=[2, 2],
            feat_extract_norm=feat_extract_norm,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        output_hidden_states: bool,
        **kwargs,
    ) -> SimpleNamespace:
        del output_hidden_states
        self.forward_kwargs = kwargs
        batch_size = input_values.shape[0]
        hidden_states = (
            torch.zeros(batch_size, 4, 4, device=input_values.device),
            torch.ones(batch_size, 4, 4, device=input_values.device),
        )
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states,
        )


def test_wav2vec_extractor_freezes_model_and_returns_selected_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeSSLModel()
    monkeypatch.setattr(
        "models.ssl_extractor.AutoModel.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )
    extractor = Wav2Vec2Extractor(
        model_name="fake",
        frozen=True,
        layer=0,
    )

    output = extractor(
        torch.randn(2, 20),
        torch.tensor([20, 12]),
    )

    assert extractor.output_dim == 4
    assert all(not parameter.requires_grad for parameter in fake_model.parameters())
    assert output.values.shape == (2, 4, 4)
    assert output.values.sum().item() == 0
    assert output.lengths.tolist() == [4, 2]
    assert output.metadata["layer"] == 0


def test_ssl_registry_builds_wav2vec2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "models.ssl_extractor.AutoModel.from_pretrained",
        lambda *args, **kwargs: _FakeSSLModel(),
    )

    extractor = build_ssl(
        {
            "type": "wav2vec2",
            "model_name": "fake",
            "frozen": True,
            "layer": -1,
        }
    )

    assert isinstance(extractor, Wav2Vec2Extractor)


def test_waveform_normalization_uses_only_valid_samples() -> None:
    waveforms = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ]
    )
    lengths = torch.tensor([3, 5])

    normalized = normalize_waveforms(waveforms, lengths)

    assert torch.allclose(normalized[0, 3:], torch.zeros(2))
    assert torch.allclose(normalized[0, :3].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized[1].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        normalized[0, :3].var(unbiased=False),
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_group_norm_wav2vec_omits_attention_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeSSLModel(feat_extract_norm="group")
    monkeypatch.setattr(
        "models.ssl_extractor.AutoModel.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )
    extractor = Wav2Vec2Extractor(model_name="fake")

    extractor(torch.randn(2, 20), torch.tensor([20, 12]))

    assert "attention_mask" not in fake_model.forward_kwargs


def test_layer_norm_wav2vec_receives_attention_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeSSLModel(feat_extract_norm="layer")
    monkeypatch.setattr(
        "models.ssl_extractor.AutoModel.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )
    extractor = Wav2Vec2Extractor(model_name="fake")

    extractor(torch.randn(2, 20), torch.tensor([20, 12]))

    assert fake_model.forward_kwargs["attention_mask"].shape == (2, 20)
