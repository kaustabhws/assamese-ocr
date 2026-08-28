from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk

from axomiya_ocr.training.metrics import evaluate_pairs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Evaluate stored bootstrap predictions in an aligned OCR dataset"
    )
    parser.add_argument("--dataset", default="data/processed/wikisource_assamese")
    parser.add_argument(
        "--output", default="artifacts/wikisource_bootstrap_metrics.json"
    )
    args = parser.parse_args()
    dataset = load_from_disk(args.dataset)
    report = {}
    for split, rows in dataset.items():
        scores = rows["alignment_score"]
        report[split] = {
            "documents": len(set(rows["source_document"])),
            "samples": len(rows),
            "minimum_alignment_score": min(scores, default=0.0),
            "mean_alignment_score": sum(scores) / max(1, len(scores)),
            "metrics": evaluate_pairs(
                rows["text"], rows["bootstrap_prediction"]
            ).to_dict(),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
