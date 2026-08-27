from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .text import ASSAMESE_OCR_REPERTOIRE, normalize_label


@dataclass(frozen=True)
class Vocabulary:
    """Character vocabulary with CTC blank fixed at index zero."""

    characters: tuple[str, ...]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        required: Sequence[str] = ASSAMESE_OCR_REPERTOIRE,
    ) -> Vocabulary:
        chars = {char for text in texts for char in normalize_label(text)}
        chars.update(required)
        chars.discard("")
        return cls(tuple(sorted(chars, key=ord)))

    @property
    def size(self) -> int:
        return len(self.characters) + 1

    @property
    def char_to_id(self) -> dict[str, int]:
        return {char: index + 1 for index, char in enumerate(self.characters)}

    @property
    def id_to_char(self) -> dict[int, str]:
        return {index + 1: char for index, char in enumerate(self.characters)}

    @property
    def sha256(self) -> str:
        payload = "\n".join(self.characters).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def encode(self, text: str) -> list[int]:
        mapping = self.char_to_id
        normalized = normalize_label(text)
        missing = sorted({char for char in normalized if char not in mapping}, key=ord)
        if missing:
            rendered = ", ".join(f"{char!r} U+{ord(char):04X}" for char in missing)
            raise ValueError(f"Characters absent from training vocabulary: {rendered}")
        return [mapping[char] for char in normalized]

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        mapping = self.id_to_char
        output: list[str] = []
        previous: int | None = None
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id != 0 and token_id != previous:
                output.append(mapping.get(token_id, ""))
            previous = token_id
        return "".join(output)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "blank_id": 0,
            "normalization": "NFC",
            "language": "as",
            "characters": list(self.characters),
            "sha256": self.sha256,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls(tuple(payload["characters"]))
        expected = payload.get("sha256")
        if expected and expected != vocab.sha256:
            raise ValueError(f"Vocabulary checksum mismatch in {path}")
        if payload.get("blank_id", 0) != 0:
            raise ValueError("Only CTC blank_id=0 is supported")
        return vocab
