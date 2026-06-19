from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


DEFAULT_TOKENS = ["<blank>", "<unk>", " ", "'", *list("abcdefghijklmnopqrstuvwxyz")]
_INVALID_CHARACTERS = re.compile(r"[^a-z' ]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TokenizerState:
    tokens: list[str]


class CharacterCTCTokenizer:
    def __init__(self, tokens: Sequence[str] | None = None) -> None:
        self.tokens = list(tokens or DEFAULT_TOKENS)
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        if "<blank>" not in self.tokens or "<unk>" not in self.tokens:
            raise ValueError("Tokenizer vocabulary requires <blank> and <unk>")
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.blank_id = self.token_to_id["<blank>"]
        self.unk_id = self.token_to_id["<unk>"]

    def __len__(self) -> int:
        return len(self.tokens)

    def normalize(self, text: str) -> str:
        normalized = text.lower()
        normalized = _INVALID_CHARACTERS.sub("", normalized)
        normalized = _WHITESPACE.sub(" ", normalized)
        return normalized.strip()

    def encode(self, text: str) -> list[int]:
        normalized = self.normalize(text)
        return [self.token_to_id.get(character, self.unk_id) for character in normalized]

    def decode_targets(self, token_ids: Iterable[int]) -> str:
        characters = [
            self.tokens[token_id]
            for token_id in token_ids
            if token_id not in {self.blank_id, self.unk_id}
            and 0 <= token_id < len(self.tokens)
        ]
        return "".join(characters).strip()

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        collapsed: list[int] = []
        previous: int | None = None
        for token_id in token_ids:
            if token_id != previous and token_id != self.blank_id:
                collapsed.append(token_id)
            previous = token_id
        return self.decode_targets(collapsed)

    def state_dict(self) -> dict[str, list[str]]:
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Sequence[str]],
    ) -> "CharacterCTCTokenizer":
        tokens = state.get("tokens")
        if tokens is None:
            raise ValueError("Tokenizer state is missing tokens")
        return cls(tokens=tokens)
