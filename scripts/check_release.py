from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check measurable recognizer release gates")
    parser.add_argument("--metrics", default="artifacts/recognizer/test_metrics.json")
    parser.add_argument("--metadata", default="artifacts/recognizer/assamese_recognizer.int8.json")
    parser.add_argument("--max-cer", type=float, default=0.03)
    parser.add_argument("--max-size-mb", type=float, default=15.0)
    args = parser.parse_args()
    metrics_payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]
    size_mb = metadata["int8_bytes"] / 1024 / 1024
    checks = {
        "cer": {"value": metrics["cer"], "limit": args.max_cer, "passed": metrics["cer"] <= args.max_cer},
        "size_mb": {"value": size_mb, "limit": args.max_size_mb, "passed": size_mb <= args.max_size_mb},
        "ra_coverage": {
            "value": metrics["assamese_specific_recall"]["ৰ"]["reference_count"],
            "limit": 100,
            "passed": metrics["assamese_specific_recall"]["ৰ"]["reference_count"] >= 100,
        },
        "wa_coverage": {
            "value": metrics["assamese_specific_recall"]["ৱ"]["reference_count"],
            "limit": 100,
            "passed": metrics["assamese_specific_recall"]["ৱ"]["reference_count"] >= 100,
        },
    }
    result = {"passed": all(item["passed"] for item in checks.values()), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

