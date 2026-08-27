from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .schema import BoundingBox, Region

HERON_LABELS = (
    "caption",
    "footnote",
    "formula",
    "list_item",
    "page_footer",
    "page_header",
    "picture",
    "section_header",
    "table",
    "text",
    "title",
    "document_index",
    "code",
    "checkbox_selected",
    "checkbox_unselected",
    "form",
    "key_value_region",
)


class HeronLayoutDetector:
    """Thin ONNX adapter for Docling Heron's exported inference contract."""

    def __init__(self, model_path: str | Path, score_threshold: float = 0.35) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.score_threshold = score_threshold

    def predict(self, image: Image.Image) -> list[Region]:
        width, height = image.size
        resized = image.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)[None, ...]
        sizes = np.asarray([[width, height]], dtype=np.int64)
        outputs = self.session.run(None, {"images": pixels, "orig_target_sizes": sizes})
        names = [output.name for output in self.session.get_outputs()]
        result = dict(zip(names, outputs, strict=True))
        labels = np.asarray(result["labels"])[0]
        boxes = np.asarray(result["boxes"])[0]
        scores = np.asarray(result["scores"])[0]
        regions: list[Region] = []
        for index, (label_id, box, score) in enumerate(
            zip(labels, boxes, scores, strict=True)
        ):
            score = float(score)
            if score < self.score_threshold:
                continue
            x0, y0, x1, y1 = [float(value) for value in box]
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(width), x1), min(float(height), y1)
            if x1 <= x0 or y1 <= y0:
                continue
            label_index = int(label_id)
            label = HERON_LABELS[label_index] if label_index < len(HERON_LABELS) else "unknown"
            regions.append(
                Region(
                    id=f"region-{index:04d}",
                    label=label,
                    bbox=BoundingBox(x0, y0, x1, y1),
                    confidence=score,
                )
            )
        return regions
