from __future__ import annotations

from pathlib import Path

import pytest

from inference import discover_audio_files


def test_discover_audio_files_returns_supported_files_in_stable_order(
    tmp_path: Path,
) -> None:
    for name in ["b.WAV", "a.flac", "d.ogg", "c.mp3", "notes.txt"]:
        (tmp_path / name).write_bytes(b"test")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.wav").write_bytes(b"test")

    files = discover_audio_files(tmp_path)

    assert [path.name for path in files] == ["a.flac", "b.WAV", "c.mp3", "d.ogg"]


def test_discover_audio_files_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No supported audio files"):
        discover_audio_files(tmp_path)
