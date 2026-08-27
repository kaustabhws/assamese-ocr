from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .text import ctc_required_steps, normalize_label
from .vocab import Vocabulary


def _round_up(value: int, multiple: int) -> int:
    return max(multiple, int(math.ceil(value / multiple) * multiple))


def degrade_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """Light scan/camera degradation without changing the transcription."""
    image = image.convert("L")
    if rng.random() < 0.55:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 1.45))
    if rng.random() < 0.35:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.2))
    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 1.1)))
    if rng.random() < 0.25:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0.0, rng.uniform(1.5, 8.0), size=array.shape
        )
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="L")
    if rng.random() < 0.2:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(35, 90))
        image = Image.open(buffer).convert("L")
    return image


def prepare_image(
    image: Image.Image,
    *,
    height: int,
    max_width: int,
    min_width: int,
    min_ctc_steps: int = 1,
    stride: int = 4,
) -> tuple[np.ndarray, int]:
    """Resize a text crop, preserve aspect ratio, and return ink-positive CHW data."""
    image = image.convert("L")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image has invalid dimensions")
    aspect_width = int(round(image.width * height / image.height))
    required_width = min_ctc_steps * stride
    target_width = _round_up(max(min_width, aspect_width, required_width), stride)
    if required_width > max_width:
        raise ValueError(
            f"Label needs at least {required_width}px for CTC but max_width={max_width}"
        )
    resized_width = min(max_width, max(stride, aspect_width))
    image = image.resize((resized_width, height), Image.Resampling.BILINEAR)
    canvas_width = min(max_width, target_width)
    canvas = Image.new("L", (canvas_width, height), color=255)
    if resized_width > canvas_width:
        image = image.resize((canvas_width, height), Image.Resampling.BILINEAR)
    canvas.paste(image, (0, 0))
    array = 1.0 - np.asarray(canvas, dtype=np.float32) / 255.0
    return array[None, :, :], canvas_width


@dataclass
class CTCCollator:
    vocab: Vocabulary
    image_height: int = 48
    min_image_width: int = 32
    max_image_width: int = 768
    augment: bool = False
    seed: int = 1337

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        prepared: list[np.ndarray] = []
        widths: list[int] = []
        labels: list[str] = []
        encoded: list[list[int]] = []
        for index, example in enumerate(examples):
            text = normalize_label(example["text"])
            image = example["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            if self.augment:
                image = degrade_image(image, random.Random(self.seed + index + random.randrange(2**24)))
            tokens = self.vocab.encode(text)
            array, width = prepare_image(
                image,
                height=self.image_height,
                min_width=self.min_image_width,
                max_width=self.max_image_width,
                min_ctc_steps=ctc_required_steps(text),
            )
            prepared.append(array)
            widths.append(width)
            labels.append(text)
            encoded.append(tokens)

        batch_width = max(widths)
        images = np.zeros((len(prepared), 1, self.image_height, batch_width), dtype=np.float32)
        for index, array in enumerate(prepared):
            images[index, :, :, : array.shape[-1]] = array
        targets = [token for item in encoded for token in item]
        return {
            "images": torch.from_numpy(images),
            "input_widths": torch.tensor(widths, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_lengths": torch.tensor([len(item) for item in encoded], dtype=torch.long),
            "texts": labels,
        }

