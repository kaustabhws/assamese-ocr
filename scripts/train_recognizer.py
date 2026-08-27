from __future__ import annotations

import argparse
import json

from axomiya_ocr.training.config import load_config
from axomiya_ocr.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Assamese CRNN-CTC recognizer")
    parser.add_argument("--config", default="configs/recognizer.yaml")
    parser.add_argument("--resume", help="Resume from a last.pt checkpoint")
    args = parser.parse_args()
    print(json.dumps(train(load_config(args.config), args.resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
