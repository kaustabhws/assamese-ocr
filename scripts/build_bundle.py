from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_model(source: Path, destination: Path) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)
    return {
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a versioned portable inference bundle")
    parser.add_argument(
        "--recognizer", default="artifacts/recognizer/assamese_recognizer.int8.onnx"
    )
    parser.add_argument("--recognizer-metadata", default="artifacts/recognizer/assamese_recognizer.int8.json")
    parser.add_argument("--layout", default="artifacts/layout/docling-layout-heron-int8.onnx")
    parser.add_argument("--output", default="dist/axomiya-ocr-0.1.0")
    args = parser.parse_args()

    import rapidocr

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rapidocr_models = Path(rapidocr.__file__).parent / "models"
    detector_candidates = sorted(rapidocr_models.glob("*det*small*.onnx"))
    if not detector_candidates:
        detector_candidates = sorted(rapidocr_models.glob("*det*.onnx"))
    if not detector_candidates:
        raise FileNotFoundError(f"No RapidOCR detector model found in {rapidocr_models}")

    files = {
        "recognizer": copy_model(Path(args.recognizer), output / "assamese_recognizer.int8.onnx"),
        "recognizer_metadata": copy_model(
            Path(args.recognizer_metadata), output / "assamese_recognizer.int8.json"
        ),
        "layout": copy_model(Path(args.layout), output / "layout_heron.int8.onnx"),
        "text_detector": copy_model(detector_candidates[-1], output / "text_detector.onnx"),
    }
    manifest = {
        "bundle_format": 1,
        "language": "as",
        "normalization": "NFC",
        "rapidocr_version": importlib.metadata.version("rapidocr"),
        "files": files,
        "licenses": {
            "project": "Apache-2.0",
            "layout": "Apache-2.0 (inherited from Docling Heron)",
            "text_detector": "Apache-2.0 (RapidOCR/PaddleOCR)",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
