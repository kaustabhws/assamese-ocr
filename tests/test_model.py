import pytest

torch = pytest.importorskip("torch")

from axomiya_ocr.models.recognizer import (  # noqa: E402
    AssameseCRNN,
    RecognizerConfig,
    output_lengths,
)


def test_recognizer_dynamic_width_contract() -> None:
    model = AssameseCRNN(
        RecognizerConfig(num_classes=80, cnn_channels=96, rnn_hidden=32, rnn_layers=1)
    ).eval()
    for width in (64, 128, 260):
        output = model(torch.zeros(2, 1, 48, width))
        assert output.shape == (2, width // 4, 80)
    assert output_lengths(torch.tensor([64, 260])).tolist() == [16, 65]
