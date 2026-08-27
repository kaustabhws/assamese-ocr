from __future__ import annotations

import argparse
import json

from axomiya_ocr.data.mozhi import prepare_mozhi


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare only the Assamese Mozhi OCR split")
    parser.add_argument("--output", default="data/processed/mozhi_assamese")
    parser.add_argument("--max-per-split", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_mozhi(
        args.output,
        max_per_split=args.max_per_split or None,
        num_proc=args.num_proc,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
