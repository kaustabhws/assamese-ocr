from __future__ import annotations

import argparse
import sys
from pathlib import Path

from axomiya_ocr.export.document import (
    save_html,
    save_json,
    save_searchable_pdf,
    save_text,
)
from axomiya_ocr.inference.document import DocumentOCR
from axomiya_ocr.inference.recognizer import ONNXRecognizer
from axomiya_ocr.inference.text_detector import RapidTextDetector
from axomiya_ocr.layout.detector import HeronLayoutDetector

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = REPOSITORY_ROOT / "dist" / "axomiya-ocr-0.2.0"


def _required_file(bundle: Path, filename: str) -> Path:
    path = bundle / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required model file not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run layout-preserving Assamese OCR on an image or PDF."
    )
    parser.add_argument("input", type=Path, help="Input image or PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: ocr_output/<input-name>)",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help=f"Model bundle directory (default: {DEFAULT_BUNDLE})",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    parser.add_argument(
        "--font",
        type=Path,
        help="Optional Assamese font; also creates a searchable document.pdf",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    source = args.input.resolve()
    bundle = args.bundle.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    output = (args.output or REPOSITORY_ROOT / "ocr_output" / source.stem).resolve()
    output.mkdir(parents=True, exist_ok=True)

    recognizer_path = _required_file(bundle, "assamese_recognizer.int8.onnx")
    metadata_path = _required_file(bundle, "assamese_recognizer.int8.json")
    layout_path = _required_file(bundle, "layout_heron.int8.onnx")
    text_detector_path = _required_file(bundle, "text_detector.onnx")

    print(f"Loading models from {bundle}")
    pipeline = DocumentOCR(
        HeronLayoutDetector(layout_path),
        ONNXRecognizer(recognizer_path, metadata_path),
        RapidTextDetector(text_detector_path),
    )
    print(f"Processing {source}")
    document, page_images = pipeline.process(source, dpi=args.dpi)

    save_text(document, output / "text.txt")
    save_json(document, output / "document.json")
    save_html(document, page_images, output / "document.html")
    if args.font:
        font = args.font.resolve()
        if not font.is_file():
            raise FileNotFoundError(f"Font file not found: {font}")
        save_searchable_pdf(document, page_images, output / "document.pdf", font)

    lines = sum(len(region.lines) for page in document.pages for region in page.regions)
    print(f"Done: {len(document.pages)} page(s), {lines} text line(s)")
    print(f"Text: {output / 'text.txt'}")
    print(f"Layout JSON: {output / 'document.json'}")
    print(f"Searchable HTML: {output / 'document.html'}")


if __name__ == "__main__":
    main()
