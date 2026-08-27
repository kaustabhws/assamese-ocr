from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from axomiya_ocr.layout.reading_order import resolve_reading_order
from axomiya_ocr.layout.schema import BoundingBox, Document, Page, Region, TextLine

from .text_detector import RapidTextDetector, rectify_crop


class LayoutDetector(Protocol):
    def predict(self, image: Image.Image) -> list[Region]: ...


class TextDetector(Protocol):
    def predict(self, image: Image.Image) -> list[list[list[float]]]: ...


class TextRecognizer(Protocol):
    def predict(self, image: Image.Image) -> tuple[str, float]: ...


def rasterize_document(path: str | Path, dpi: int = 200) -> Iterator[Image.Image]:
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        yield Image.open(path).convert("RGB")
        return
    import fitz

    document = fitz.open(path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for pdf_page in document:
        pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)
        yield Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _polygon_bbox(polygon: list[list[float]]) -> BoundingBox:
    points = np.asarray(polygon)
    return BoundingBox(
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _contains_center(region: Region, line_box: BoundingBox) -> bool:
    center_x = (line_box.x0 + line_box.x1) / 2
    center_y = (line_box.y0 + line_box.y1) / 2
    return (
        region.bbox.x0 <= center_x <= region.bbox.x1
        and region.bbox.y0 <= center_y <= region.bbox.y1
    )


class DocumentOCR:
    def __init__(
        self,
        layout_detector: LayoutDetector,
        recognizer: TextRecognizer,
        text_detector: TextDetector | None = None,
    ) -> None:
        self.layout_detector = layout_detector
        self.text_detector = text_detector or RapidTextDetector()
        self.recognizer = recognizer

    def process(self, source_path: str | Path, dpi: int = 200) -> tuple[Document, list[Image.Image]]:
        pages: list[Page] = []
        page_images: list[Image.Image] = []
        for page_number, image in enumerate(rasterize_document(source_path, dpi=dpi), start=1):
            page_images.append(image)
            regions = self.layout_detector.predict(image)
            polygons = self.text_detector.predict(image)
            for line_index, polygon in enumerate(polygons):
                line_box = _polygon_bbox(polygon)
                crop = rectify_crop(image, polygon)
                text, confidence = self.recognizer.predict(crop)
                line = TextLine(
                    id=f"p{page_number}-line-{line_index:05d}",
                    bbox=line_box,
                    polygon=polygon,
                    text=text,
                    confidence=confidence,
                )
                containing = [region for region in regions if _contains_center(region, line_box)]
                if containing:
                    region = min(containing, key=lambda item: item.bbox.width * item.bbox.height)
                    region.lines.append(line)
                else:
                    regions.append(
                        Region(
                            id=f"region-fallback-{line_index:04d}",
                            label="text",
                            bbox=line_box,
                            confidence=0.0,
                            lines=[line],
                        )
                    )
            regions = resolve_reading_order(regions, image.width)
            pages.append(Page(page_number, image.width, image.height, regions))
        document = Document(source=str(source_path), pages=pages)
        return document, page_images
