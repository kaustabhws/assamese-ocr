from __future__ import annotations

import argparse
import json

from axomiya_ocr.export.onnx import export_recognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify the Assamese recognizer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/recognizer/assamese_recognizer.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    metadata = export_recognizer(
        args.checkpoint, args.output, opset=args.opset, quantize=args.quantize
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

