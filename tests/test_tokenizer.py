from __future__ import annotations

from utils.tokenizer import CharacterCTCTokenizer


def test_tokenizer_normalizes_and_round_trips_targets() -> None:
    tokenizer = CharacterCTCTokenizer()

    token_ids = tokenizer.encode("Hello,   world's!")

    assert tokenizer.decode_targets(token_ids) == "hello world's"


def test_ctc_decode_collapses_repeats_and_removes_blank() -> None:
    tokenizer = CharacterCTCTokenizer()
    h_id = tokenizer.token_to_id["h"]
    i_id = tokenizer.token_to_id["i"]

    decoded = tokenizer.decode_ctc(
        [h_id, h_id, tokenizer.blank_id, i_id, i_id]
    )

    assert decoded == "hi"


def test_tokenizer_state_round_trip() -> None:
    tokenizer = CharacterCTCTokenizer()

    restored = CharacterCTCTokenizer.from_state_dict(tokenizer.state_dict())

    assert restored.tokens == tokenizer.tokens
    assert restored.blank_id == tokenizer.blank_id
