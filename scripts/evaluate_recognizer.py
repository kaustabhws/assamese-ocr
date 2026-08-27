from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from axomiya_ocr.data.dataset import HFDiskOCRDataset
from axomiya_ocr.data.image import CTCCollator
from axomiya_ocr.training.engine import evaluate, model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on an untouched split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/processed/mozhi_assamese")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--output", default="artifacts/recognizer/test_metrics.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, checkpoint = model_from_checkpoint(args.checkpoint, device)
    data_config = checkpoint["training_config"]["data"]
    dataset = HFDiskOCRDataset(args.dataset, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, __import__("os").cpu_count() or 1),
        collate_fn=CTCCollator(
            vocab=vocab,
            image_height=int(data_config["image_height"]),
            min_image_width=int(data_config["min_image_width"]),
            max_image_width=int(data_config["max_image_width"]),
        ),
    )
    metrics, predictions = evaluate(model, loader, vocab, device)
    payload = {"split": args.split, "metrics": metrics, "sample_predictions": predictions}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

