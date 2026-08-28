from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from axomiya_ocr.data.image import prepare_image
from axomiya_ocr.data.vocab import Vocabulary


class ONNXRecognizer:
    def __init__(self, model_path: str | Path, metadata_path: str | Path | None = None) -> None:
        import onnxruntime as ort

        model_path = Path(model_path)
        metadata_path = Path(metadata_path) if metadata_path else model_path.with_suffix(".json")
        if not metadata_path.exists() and model_path.name.endswith(".int8.onnx"):
            metadata_path = model_path.with_name(model_path.name.replace(".int8.onnx", ".json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.vocab = Vocabulary(tuple(metadata["vocab"]["characters"]))
        self.height = int(metadata["input"]["height"])
        self.max_width = int(metadata["input"].get("max_width", 768))
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def predict(self, image: Image.Image, max_width: int | None = None) -> tuple[str, float]:
        max_width = max_width or self.max_width
        array, width = prepare_image(
            image,
            height=self.height,
            min_width=32,
            max_width=max_width,
            min_ctc_steps=1,
        )
        logits = self.session.run(["logits"], {"images": array[None, ...]})[0][0]
        logits = logits[: width // 4]
        logits -= logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        best = probabilities.argmax(axis=-1)
        text = self.vocab.decode_ctc(best)
        keep = np.logical_and(best != 0, np.concatenate(([True], best[1:] != best[:-1])))
        confidence = float(probabilities[np.arange(len(best)), best][keep].mean()) if keep.any() else 0.0
        return text, confidence
