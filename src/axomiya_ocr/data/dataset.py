from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class HFDiskOCRDataset(Dataset[dict[str, Any]]):
    """Small PyTorch-compatible adapter around a saved Hugging Face split."""

    def __init__(self, dataset_path: str | Path, split: str) -> None:
        from datasets import load_from_disk

        dataset = load_from_disk(str(dataset_path))
        if split not in dataset:
            raise KeyError(f"Split {split!r} is absent from {dataset_path}")
        self.dataset = dataset[split]

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def texts(self) -> Sequence[str]:
        return self.dataset["text"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        return {"id": row["id"], "image": row["image"], "text": row["text"]}
