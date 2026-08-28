from __future__ import annotations

import argparse
import json

from axomiya_ocr.training.config import load_config
from axomiya_ocr.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Assamese CRNN-CTC recognizer")
    parser.add_argument("--config", default="configs/recognizer.yaml")
    parser.add_argument("--resume", help="Resume from a last.pt checkpoint")
    parser.add_argument(
        "--init-from",
        help="Warm-start model weights from a checkpoint and reset optimizer/epoch state",
    )
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")
    print(
        json.dumps(
            train(load_config(args.config), args.resume, args.init_from),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
