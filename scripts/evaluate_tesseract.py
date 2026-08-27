from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_from_disk

from axomiya_ocr.data.text import normalize_label
from axomiya_ocr.training.metrics import OCRMetrics, character_recall


def find_tesseract(explicit: str | None) -> str:
    if explicit:
        return explicit
    discovered = shutil.which("tesseract")
    candidates = [
        discovered,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Tesseract executable not found")


def recognize(executable: str, tessdata: Path, image: object) -> str:
    buffer = io.BytesIO()
    image.convert("L").save(buffer, format="PNG")
    process = subprocess.run(
        [
            executable,
            "stdin",
            "stdout",
            "--tessdata-dir",
            str(tessdata),
            "-l",
            "asm",
            "--psm",
            "7",
        ],
        input=buffer.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return normalize_label(process.stdout.decode("utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the official Assamese Tesseract baseline")
    parser.add_argument("--dataset", default="data/processed/mozhi_assamese")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--profile", choices=("fast", "best"), default="fast")
    parser.add_argument("--tesseract")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", default="artifacts/tesseract/baseline_metrics.json")
    args = parser.parse_args()
    executable = find_tesseract(args.tesseract)
    tessdata = Path("artifacts/tesseract") / args.profile
    if not (tessdata / "asm.traineddata").exists():
        raise FileNotFoundError("Run scripts/download_tesseract_baselines.py first")
    dataset = load_from_disk(args.dataset)[args.split]
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    metrics = OCRMetrics()
    references: list[str] = []
    hypotheses: list[str] = []
    started = time.perf_counter()
    def evaluate_index(index: int) -> tuple[str, str]:
        row = dataset[index]
        return normalize_label(row["text"]), recognize(executable, tessdata, row["image"])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(evaluate_index, range(len(dataset)))
        for index, (reference, hypothesis) in enumerate(results):
            metrics.update(reference, hypothesis)
            references.append(reference)
            hypotheses.append(hypothesis)
            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1}/{len(dataset)}", flush=True)
    payload = {
        "engine": "tesseract",
        "language": "asm",
        "profile": args.profile,
        "split": args.split,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics.to_dict(),
        "assamese_specific_recall": {
            char: character_recall(references, hypotheses, char) for char in ("ৰ", "ৱ")
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
