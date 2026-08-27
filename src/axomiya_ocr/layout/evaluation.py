from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from axomiya_ocr.training.metrics import OCRMetrics, character_recall


def bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    x0 = max(left["x0"], right["x0"])
    y0 = max(left["y0"], right["y0"])
    x1 = min(left["x1"], right["x1"])
    y1 = min(left["y1"], right["y1"])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left["x1"] - left["x0"]) * max(0.0, left["y1"] - left["y0"])
    right_area = max(0.0, right["x1"] - right["x0"]) * max(
        0.0, right["y1"] - right["y0"]
    )
    return intersection / max(1e-9, left_area + right_area - intersection)


@dataclass
class DetectionCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def to_dict(self) -> dict[str, float | int]:
        precision = self.true_positive / max(1, self.true_positive + self.false_positive)
        recall = self.true_positive / max(1, self.true_positive + self.false_negative)
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-9, precision + recall),
        }


def match_regions(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for expected_index, expected_region in enumerate(expected):
        for predicted_index, predicted_region in enumerate(predicted):
            if expected_region["label"] != predicted_region["label"]:
                continue
            overlap = bbox_iou(expected_region["bbox"], predicted_region["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, expected_index, predicted_index))
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, expected_index, predicted_index in sorted(candidates, reverse=True):
        if expected_index in matched_expected or predicted_index in matched_predicted:
            continue
        matches.append((expected_index, predicted_index))
        matched_expected.add(expected_index)
        matched_predicted.add(predicted_index)
    return matches


def _page_text(page: dict[str, Any]) -> str:
    regions = sorted(page["regions"], key=lambda region: region["order"])
    lines = [
        line["text"]
        for region in regions
        for line in sorted(region.get("lines", []), key=lambda item: item["order"])
    ]
    return "\n".join(lines)


def evaluate_documents(
    expected: dict[str, Any],
    predicted: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    expected_pages = {page["number"]: page for page in expected["pages"]}
    predicted_pages = {page["number"]: page for page in predicted["pages"]}
    if set(expected_pages) != set(predicted_pages):
        raise ValueError("Expected and predicted documents must contain the same page numbers")

    overall = DetectionCounts()
    per_label: dict[str, DetectionCounts] = defaultdict(DetectionCounts)
    ocr = OCRMetrics()
    references: list[str] = []
    hypotheses: list[str] = []
    concordant = comparable = 0
    for page_number in sorted(expected_pages):
        expected_page = expected_pages[page_number]
        predicted_page = predicted_pages[page_number]
        expected_regions = expected_page["regions"]
        predicted_regions = predicted_page["regions"]
        matches = match_regions(expected_regions, predicted_regions, iou_threshold)
        overall.true_positive += len(matches)
        overall.false_negative += len(expected_regions) - len(matches)
        overall.false_positive += len(predicted_regions) - len(matches)
        matched_expected = {left for left, _ in matches}
        matched_predicted = {right for _, right in matches}
        for index, region in enumerate(expected_regions):
            if index not in matched_expected:
                per_label[region["label"]].false_negative += 1
        for index, region in enumerate(predicted_regions):
            if index not in matched_predicted:
                per_label[region["label"]].false_positive += 1
        for expected_index, _ in matches:
            per_label[expected_regions[expected_index]["label"]].true_positive += 1

        ordered_matches = sorted(
            matches, key=lambda pair: expected_regions[pair[0]]["order"]
        )
        predicted_orders = [predicted_regions[right]["order"] for _, right in ordered_matches]
        for left in range(len(predicted_orders)):
            for right in range(left + 1, len(predicted_orders)):
                comparable += 1
                concordant += predicted_orders[left] < predicted_orders[right]

        reference, hypothesis = _page_text(expected_page), _page_text(predicted_page)
        references.append(reference)
        hypotheses.append(hypothesis)
        ocr.update(reference, hypothesis)

    return {
        "iou_threshold": iou_threshold,
        "layout": overall.to_dict(),
        "layout_by_class": {label: counts.to_dict() for label, counts in sorted(per_label.items())},
        "reading_order_pair_accuracy": concordant / max(1, comparable),
        "reading_order_pairs": comparable,
        "ocr": ocr.to_dict(),
        "assamese_specific_recall": {
            char: character_recall(references, hypotheses, char) for char in ("ৰ", "ৱ")
        },
    }

