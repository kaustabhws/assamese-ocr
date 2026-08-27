from __future__ import annotations

from collections.abc import Sequence

from .schema import BoundingBox, Region


def _horizontal_overlap(a: BoundingBox, b: BoundingBox) -> float:
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    return overlap / max(1.0, min(a.width, b.width))


def _column_groups(regions: Sequence[Region], page_width: float) -> list[list[Region]]:
    """Cluster a horizontal page band into left-to-right columns."""
    if not regions:
        return []
    sorted_regions = sorted(regions, key=lambda region: (region.bbox.x0, region.bbox.y0))
    columns: list[list[Region]] = []
    for region in sorted_regions:
        best_index = -1
        best_overlap = 0.0
        for index, column in enumerate(columns):
            envelope = BoundingBox(
                min(item.bbox.x0 for item in column),
                min(item.bbox.y0 for item in column),
                max(item.bbox.x1 for item in column),
                max(item.bbox.y1 for item in column),
            )
            overlap = _horizontal_overlap(region.bbox, envelope)
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index >= 0 and best_overlap >= 0.25:
            columns[best_index].append(region)
        else:
            columns.append([region])
    columns.sort(key=lambda column: min(item.bbox.x0 for item in column) / max(1.0, page_width))
    for column in columns:
        column.sort(key=lambda item: (item.bbox.y0, item.bbox.x0))
    return columns


def resolve_reading_order(
    regions: Sequence[Region], page_width: float, full_width_ratio: float = 0.62
) -> list[Region]:
    """Order headers/full-width separators and multi-column content deterministically."""
    wide = sorted(
        [region for region in regions if region.bbox.width / max(1.0, page_width) >= full_width_ratio],
        key=lambda item: (item.bbox.y0, item.bbox.x0),
    )
    narrow = [region for region in regions if region not in wide]

    ordered: list[Region] = []
    lower_bound = float("-inf")
    for separator in wide:
        band = [
            region
            for region in narrow
            if region.bbox.y0 >= lower_bound and region.bbox.y0 < separator.bbox.y0
        ]
        for column in _column_groups(band, page_width):
            ordered.extend(column)
        narrow = [region for region in narrow if region not in band]
        ordered.append(separator)
        lower_bound = separator.bbox.y1

    tail = [region for region in narrow if region.bbox.y0 >= lower_bound]
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

