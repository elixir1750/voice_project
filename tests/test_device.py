from __future__ import annotations

import pytest
import torch

from utils.device import resolve_device, supports_amp


def test_explicit_cpu_is_always_available() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert supports_amp(torch.device("cpu")) is False


def test_auto_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("cuda")


def test_auto_device_uses_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("mps")


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")


def test_explicit_unavailable_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_unknown_device_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_device("tpu")
