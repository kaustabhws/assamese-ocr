import numpy as np

from axomiya_ocr.inference.recognizer import _right_pad


def test_right_pad_preserves_content_and_uses_white_image_value() -> None:
    array = np.ones((1, 48, 32), dtype=np.float32)
    padded = _right_pad(array, 64)
    assert padded.shape == (1, 48, 64)
    np.testing.assert_array_equal(padded[:, :, :32], array)
    assert np.count_nonzero(padded[:, :, 32:]) == 0
