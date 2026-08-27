from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="axomiya-ocr")
    parser.add_argument(
        "command",
        choices=("prepare", "train", "evaluate", "export", "run"),
        help="Workflow command; remaining flags are passed to its script",
    )
    args, remaining = parser.parse_known_args()
    scripts = {
        "prepare": "prepare_mozhi.py",
        "train": "train_recognizer.py",
        "evaluate": "evaluate_recognizer.py",
        "export": "export_recognizer.py",
        "run": "run_ocr.py",
    }
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call([sys.executable, str(root / "scripts" / scripts[args.command]), *remaining]))


if __name__ == "__main__":
    main()

