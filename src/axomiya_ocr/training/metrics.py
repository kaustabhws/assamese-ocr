from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, start=1):
        current = [row_index]
        for col_index, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[col_index - 1] + (ref_item != hyp_item)
            insertion = current[col_index - 1] + 1
            deletion = previous[col_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


@dataclass
class OCRMetrics:
    samples: int = 0
    char_errors: int = 0
    characters: int = 0
    word_errors: int = 0
    words: int = 0
    exact: int = 0

    def update(self, reference: str, hypothesis: str) -> None:
        self.samples += 1
        self.char_errors += edit_distance(reference, hypothesis)
        self.characters += len(reference)
        self.word_errors += edit_distance(reference.split(), hypothesis.split())
        self.words += len(reference.split())
        self.exact += reference == hypothesis

    def merge(self, other: OCRMetrics) -> None:
        for field_name in asdict(self):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))

    def to_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "cer": self.char_errors / max(1, self.characters),
            "wer": self.word_errors / max(1, self.words),
            "sequence_accuracy": self.exact / max(1, self.samples),
        }


def evaluate_pairs(references: Iterable[str], hypotheses: Iterable[str]) -> OCRMetrics:
    metrics = OCRMetrics()
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        metrics.update(reference, hypothesis)
    return metrics


def character_recall(
    references: Iterable[str], hypotheses: Iterable[str], target: str
) -> dict[str, float | int | str]:
    reference_count = 0
    matched = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_count += reference.count(target)
        matched += sum(
            ref_char == target and ref_char == hyp_char
            for ref_char, hyp_char in align_characters(reference, hypothesis)
        )
    return {
        "character": target,
        "reference_count": reference_count,
        "matched_count": matched,
        "recall": matched / max(1, reference_count),
    }


def align_characters(reference: str, hypothesis: str) -> list[tuple[str | None, str | None]]:
    """Return a deterministic minimum-edit alignment for character diagnostics."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    matrix = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for column in range(columns):
        matrix[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1]
                + (reference[row - 1] != hypothesis[column - 1]),
            )

    alignment: list[tuple[str | None, str | None]] = []
    row, column = len(reference), len(hypothesis)
    while row or column:
        if row and column:
            substitution_cost = reference[row - 1] != hypothesis[column - 1]
            if matrix[row][column] == matrix[row - 1][column - 1] + substitution_cost:
                alignment.append((reference[row - 1], hypothesis[column - 1]))
                row -= 1
                column -= 1
                continue
        if row and matrix[row][column] == matrix[row - 1][column] + 1:
            alignment.append((reference[row - 1], None))
            row -= 1
        else:
            alignment.append((None, hypothesis[column - 1]))
            column -= 1
    alignment.reverse()
    return alignment
