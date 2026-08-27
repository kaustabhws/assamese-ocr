from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def main() -> None:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description="Benchmark dynamic-width recognizer inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    sample = np.zeros((1, 1, 48, args.width), dtype=np.float32)
    for _ in range(args.warmup):
        session.run(None, {"images": sample})
    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, {"images": sample})
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    report = {
        "model": str(Path(args.model)),
        "model_bytes": Path(args.model).stat().st_size,
        "input_shape": list(sample.shape),
        "runs": args.runs,
        "median_ms": statistics.median(timings),
        "p95_ms": timings[min(len(timings) - 1, int(len(timings) * 0.95))],
        "providers": session.get_providers(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

