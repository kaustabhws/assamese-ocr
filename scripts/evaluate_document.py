from __future__ import annotations

import argparse
import json
from pathlib import Path

from axomiya_ocr.layout.evaluation import evaluate_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate layout, reading order, and page OCR")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts/page_evaluation.json")
    args = parser.parse_args()
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    result = evaluate_documents(ground_truth, prediction, args.iou)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

