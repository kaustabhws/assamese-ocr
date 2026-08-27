from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class RapidTextDetector:
    """Use RapidOCR's mobile DBNet only; its multilingual recognizer is disabled."""

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self.engine = RapidOCR()

    def predict(self, image: Image.Image) -> list[list[list[float]]]:
        result = self.engine(np.asarray(image.convert("RGB")), use_det=True, use_cls=False, use_rec=False)
        boxes: Any = getattr(result, "boxes", None)
        if boxes is None and isinstance(result, tuple):
            boxes = result[0]
        if boxes is None:
            return []
        array = np.asarray(boxes, dtype=np.float32)
        if array.size == 0:
            return []
        return array.reshape(-1, 4, 2).tolist()


def rectify_crop(image: Image.Image, polygon: list[list[float]]) -> Image.Image:
    import cv2

    points = np.asarray(polygon, dtype=np.float32)
    top = np.linalg.norm(points[1] - points[0])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(
        np.asarray(image.convert("RGB")),
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if height > width * 1.5:
        crop = np.rot90(crop)
    return Image.fromarray(crop)

