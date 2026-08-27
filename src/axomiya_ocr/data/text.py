from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

ASSAMESE_CONFIG = "assamese"
ASSAMESE_WIKIPEDIA_CONFIG = "20231101.as"
ASSAMESE_BLOCK_START = 0x0980
ASSAMESE_BLOCK_END = 0x09FF
ASSAMESE_DISTINCTIVE = ("ৰ", "ৱ")
# Fixed deployment repertoire. Unicode calls the shared Eastern Nagari block
# "Bengali"; inclusion here does not imply Bengali-language training data.
ASSAMESE_OCR_REPERTOIRE = tuple(
    " !\"'()*+,-./:;?%&0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "।॥ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
    "়ািীুূৃেৈোৌ্ৎড়ঢ়য়০১২৩৪৫৬৭৮৯ৰৱ৷‌‍–—…‘’“”"
)

_WHITESPACE = re.compile(r"\s+")


def normalize_label(text: str) -> str:
    """Normalize without transliterating or changing Assamese characters."""
    if not isinstance(text, str):
        raise TypeError("OCR labels must be strings")
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", text).strip()


def is_assamese_block_char(char: str) -> bool:
    return len(char) == 1 and ASSAMESE_BLOCK_START <= ord(char) <= ASSAMESE_BLOCK_END


def is_control(char: str) -> bool:
    return unicodedata.category(char).startswith("C") and char not in {"\u200c", "\u200d"}


def assamese_script_ratio(text: str) -> float:
    letters_and_marks = [
        char for char in text if unicodedata.category(char)[0] in {"L", "M"}
    ]
    if not letters_and_marks:
        return 1.0
    in_block = sum(is_assamese_block_char(char) for char in letters_and_marks)
    return in_block / len(letters_and_marks)


def validate_assamese_label(text: str, min_script_ratio: float = 0.6) -> tuple[bool, str]:
    normalized = normalize_label(text)
    if not normalized:
        return False, "empty"
    if any(is_control(char) for char in normalized):
        return False, "control_character"
    if assamese_script_ratio(normalized) < min_script_ratio:
        return False, "low_assamese_script_ratio"
    return True, "ok"


def ctc_required_steps(text: str) -> int:
    """Minimum CTC timesteps: labels plus one blank between repeated symbols."""
    normalized = normalize_label(text)
    repeats = sum(
        left == right for left, right in zip(normalized, normalized[1:], strict=False)
    )
    return len(normalized) + repeats


def assert_assamese_source(config_name: str) -> None:
    if config_name != ASSAMESE_CONFIG:
        raise ValueError(
            f"Refusing dataset config {config_name!r}; this project only accepts "
            f"{ASSAMESE_CONFIG!r}."
        )


@dataclass(frozen=True)
class TextAudit:
    samples: int
    empty: int
    invalid: int
    with_ra: int
    with_wa: int
    characters: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_texts(texts: Iterable[str]) -> TextAudit:
    counts: Counter[str] = Counter()
    samples = empty = invalid = with_ra = with_wa = 0
    for value in texts:
        samples += 1
        normalized = normalize_label(value)
        if not normalized:
            empty += 1
        valid, _ = validate_assamese_label(normalized)
        if not valid:
            invalid += 1
        with_ra += "ৰ" in normalized
        with_wa += "ৱ" in normalized
        counts.update(normalized)
    return TextAudit(
        samples=samples,
        empty=empty,
        invalid=invalid,
        with_ra=with_ra,
        with_wa=with_wa,
        characters=dict(sorted(counts.items(), key=lambda item: ord(item[0]))),
    )
