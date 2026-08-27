from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Invalid bounding box")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class TextLine:
    id: str
    bbox: BoundingBox
    polygon: list[list[float]]
    text: str
    confidence: float
    order: int = 0


@dataclass
class Region:
    id: str
    label: str
    bbox: BoundingBox
    confidence: float
    order: int = 0
    lines: list[TextLine] = field(default_factory=list)


@dataclass
class Page:
    number: int
    width: int
    height: int
    regions: list[Region] = field(default_factory=list)


@dataclass
class Document:
    schema_version: str = "1.0"
    language: str = "as"
    source: str | None = None
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

