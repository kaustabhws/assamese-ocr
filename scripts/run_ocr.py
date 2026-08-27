from __future__ import annotations

import argparse
from pathlib import Path

from axomiya_ocr.export.document import save_html, save_json, save_searchable_pdf
from axomiya_ocr.inference.document import DocumentOCR
from axomiya_ocr.inference.recognizer import ONNXRecognizer
from axomiya_ocr.layout.detector import HeronLayoutDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layout-preserving Assamese OCR")
    parser.add_argument("input")
    parser.add_argument("--recognizer", required=True)
    parser.add_argument("--metadata")
    parser.add_argument(
        "--layout", default="artifacts/layout/docling-layout-heron-int8.onnx"
    )
    parser.add_argument("--output", default="artifacts/output")
    parser.add_argument("--font", help="Font file required only for searchable PDF output")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recognizer = ONNXRecognizer(args.recognizer, args.metadata)
    layout = HeronLayoutDetector(args.layout)
    document, page_images = DocumentOCR(layout, recognizer).process(args.input, dpi=args.dpi)
    save_json(document, output / "document.json")
    save_html(document, page_images, output / "document.html")
    if args.font:
        save_searchable_pdf(document, page_images, output / "document.pdf", args.font)
    print(output)


if __name__ == "__main__":
    main()

