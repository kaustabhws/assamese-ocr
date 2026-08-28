from __future__ import annotations

from collections.abc import Sequence

from axomiya_ocr.data.text import assamese_script_ratio

from .schema import BoundingBox, Region, TextLine

_STRUCTURAL_TEXT_LABELS = {"form", "key_value_region", "table"}
_KEEP_EMPTY_LABELS = {
    "checkbox_selected",
    "checkbox_unselected",
    "formula",
    "picture",
    "table",
}


def _box_area(box: BoundingBox) -> float:
    return max(0.0, box.width) * max(0.0, box.height)


def _intersection_over_union(a: BoundingBox, b: BoundingBox) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / max(1.0, union)


def _line_overlap(region: BoundingBox, line: BoundingBox) -> float:
    x0, y0 = max(region.x0, line.x0), max(region.y0, line.y0)
    x1, y1 = min(region.x1, line.x1), min(region.y1, line.y1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / max(1.0, _box_area(line))


def _reading_bbox(region: Region) -> BoundingBox:
    if not region.lines:
        return region.bbox
    return BoundingBox(
        min(line.bbox.x0 for line in region.lines),
        min(line.bbox.y0 for line in region.lines),
        max(line.bbox.x1 for line in region.lines),
        max(line.bbox.y1 for line in region.lines),
    )


def deduplicate_regions(regions: Sequence[Region], iou_threshold: float = 0.9) -> list[Region]:
    """Remove near-identical detector boxes, keeping the highest-confidence label."""
    kept: list[Region] = []
    for candidate in sorted(regions, key=lambda region: region.confidence, reverse=True):
        if any(
            _intersection_over_union(candidate.bbox, existing.bbox) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def assign_lines_to_regions(regions: Sequence[Region], lines: Sequence[TextLine]) -> list[Region]:
    """Attach each text line once, preferring table/form containers when present."""
    output = list(regions)
    for region in output:
        region.lines.clear()
    for line_index, line in enumerate(lines):
        candidates = [
            region for region in output if _line_overlap(region.bbox, line.bbox) >= 0.5
        ]
        structural = [
            region for region in candidates if region.label in _STRUCTURAL_TEXT_LABELS
        ]
        if structural:
            selected = min(
                structural,
                key=lambda region: (_box_area(region.bbox), -region.confidence),
            )
        elif candidates:
            selected = min(
                candidates,
                key=lambda region: (_box_area(region.bbox), -region.confidence),
            )
        else:
            selected = Region(
                id=f"region-fallback-{line_index:04d}",
                label="text",
                bbox=line.bbox,
                confidence=0.0,
            )
            output.append(selected)
        if selected.label == "picture" and assamese_script_ratio(line.text) < 0.6:
            continue
        selected.lines.append(line)
    return [region for region in output if region.lines or region.label in _KEEP_EMPTY_LABELS]


def _horizontal_overlap(a: BoundingBox, b: BoundingBox) -> float:
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    return overlap / max(1.0, min(a.width, b.width))


def _column_groups(regions: Sequence[Region], page_width: float) -> list[list[Region]]:
    """Cluster a horizontal page band into left-to-right columns."""
    if not regions:
        return []
    if len(regions) < 3:
        return [
            sorted(
                regions,
                key=lambda region: (_reading_bbox(region).y0, _reading_bbox(region).x0),
            )
        ]
    sorted_regions = sorted(
        regions, key=lambda region: (_reading_bbox(region).x0, _reading_bbox(region).y0)
    )
    columns: list[list[Region]] = []
    for region in sorted_regions:
        best_index = -1
        best_overlap = 0.0
        for index, column in enumerate(columns):
            envelope = BoundingBox(
                min(_reading_bbox(item).x0 for item in column),
                min(_reading_bbox(item).y0 for item in column),
                max(_reading_bbox(item).x1 for item in column),
                max(_reading_bbox(item).y1 for item in column),
            )
            overlap = _horizontal_overlap(_reading_bbox(region), envelope)
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index >= 0 and best_overlap >= 0.25:
            columns[best_index].append(region)
        else:
            columns.append([region])
    columns.sort(
        key=lambda column: min(_reading_bbox(item).x0 for item in column)
        / max(1.0, page_width)
    )
    for column in columns:
        column.sort(key=lambda item: (_reading_bbox(item).y0, _reading_bbox(item).x0))
    return columns


def resolve_reading_order(
    regions: Sequence[Region],
    page_width: float,
    full_width_ratio: float = 0.62,
    page_height: float | None = None,
) -> list[Region]:
    """Order headers/full-width separators and multi-column content deterministically."""
    top_cutoff = (page_height * 0.16) if page_height else float("-inf")
    top_band = sorted(
        [region for region in regions if _reading_bbox(region).y0 <= top_cutoff],
        key=lambda item: (_reading_bbox(item).y0, _reading_bbox(item).x0),
    )
    regions = [region for region in regions if region not in top_band]
    wide = sorted(
        [
            region
            for region in regions
            if _reading_bbox(region).width / max(1.0, page_width) >= full_width_ratio
        ],
        key=lambda item: (_reading_bbox(item).y0, _reading_bbox(item).x0),
    )
    narrow = [region for region in regions if region not in wide]

    ordered: list[Region] = list(top_band)
    lower_bound = float("-inf")
    for separator in wide:
        band = [
            region
            for region in narrow
            if _reading_bbox(region).y0 >= lower_bound
            and _reading_bbox(region).y0 < _reading_bbox(separator).y0
        ]
        for column in _column_groups(band, page_width):
            ordered.extend(column)
        narrow = [region for region in narrow if region not in band]
        ordered.append(separator)
        lower_bound = _reading_bbox(separator).y1

    tail = [region for region in narrow if _reading_bbox(region).y0 >= lower_bound]
    head_overlap = [region for region in narrow if region not in tail]
    for collection in (head_overlap, tail):
        for column in _column_groups(collection, page_width):
            ordered.extend(column)

    for order, region in enumerate(ordered):
        region.order = order
        region.lines.sort(key=lambda line: (line.bbox.y0, line.bbox.x0))
        for line_order, line in enumerate(region.lines):
            line.order = line_order
    return ordered

