import numpy as np
from PIL import Image

from axomiya_ocr.data.image import prepare_image


def test_prepare_image_uses_stride_aligned_width_and_ink_positive_values() -> None:
    image = Image.new("L", (101, 20), color=255)
    image.putpixel((50, 10), 0)
    array, width = prepare_image(
        image, height=48, min_width=32, max_width=768, min_ctc_steps=5
    )
    assert array.shape == (1, 48, width)
    assert width % 4 == 0
    assert np.min(array) >= 0.0
    assert np.max(array) <= 1.0

