from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


@dataclass(frozen=True)
class RecognizerConfig:
    num_classes: int
    input_channels: int = 1
    cnn_channels: int = 384
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.2


class AssameseCRNN(nn.Module):
    """Compact dynamic-width CRNN whose output width is input_width // 4."""

    horizontal_stride = 4

    def __init__(self, config: RecognizerConfig) -> None:
        super().__init__()
        self.config = config
        final_channels = config.cnn_channels
        self.visual = nn.Sequential(
            ConvBlock(config.input_channels, 48, stride=2),
            DepthwiseSeparableBlock(48, 96, stride=2),
            DepthwiseSeparableBlock(96, 160, stride=(2, 1)),
            DepthwiseSeparableBlock(160, 256, stride=(2, 1)),
            DepthwiseSeparableBlock(256, final_channels, stride=(2, 1)),
        )
        self.sequence = nn.LSTM(
            input_size=final_channels,
            hidden_size=config.rnn_hidden,
            num_layers=config.rnn_layers,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.visual(images)
        features = features.mean(dim=2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(self.dropout(sequence))

    def metadata(self) -> dict[str, object]:
        return {
            "architecture": self.__class__.__name__,
            "config": asdict(self.config),
            "horizontal_stride": self.horizontal_stride,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def output_lengths(input_widths: Tensor, horizontal_stride: int = 4) -> Tensor:
    return torch.div(input_widths, horizontal_stride, rounding_mode="floor")

