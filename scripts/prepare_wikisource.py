from __future__ import annotations

import argparse
import json
import sys

from axomiya_ocr.data.wikisource import prepare_wikisource


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Prepare aligned real Assamese scan lines from validated Wikisource pages"
    )
    parser.add_argument("--output", default="data/processed/wikisource_assamese")
    parser.add_argument(
        "--recognizer", default="artifacts/recognizer/assamese_recognizer.int8.onnx"
    )
    parser.add_argument("--metadata")
    parser.add_argument("--vocab", default="data/processed/mozhi_assamese/vocab.json")
    parser.add_argument("--train-documents", type=int, default=12)
    parser.add_argument("--validation-documents", type=int, default=2)
    parser.add_argument("--test-documents", type=int, default=2)
    parser.add_argument("--pages-per-document", type=int, default=30)
    parser.add_argument("--min-alignment-score", type=float, default=0.68)
    parser.add_argument("--max-image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache-dir", default="data/raw/wikisource_cache")
    parser.add_argument(
        "--include-djvu",
        action="store_true",
        help="Also use DjVu books; these require one Wikimedia thumbnail request per page",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unknown-license",
        action="store_true",
        help="Allow scan files without Public Domain/Creative Commons metadata",
    )
    args = parser.parse_args()
    report = prepare_wikisource(
        args.output,
        recognizer_path=args.recognizer,
        metadata_path=args.metadata,
        vocab_path=args.vocab,
        train_documents=args.train_documents,
        validation_documents=args.validation_documents,
        test_documents=args.test_documents,
        pages_per_document=args.pages_per_document,
        min_alignment_score=args.min_alignment_score,
        max_image_width=args.max_image_width,
        image_height=args.image_height,
        seed=args.seed,
        overwrite=args.overwrite,
        allow_unknown_license=args.allow_unknown_license,
        cache_dir=args.cache_dir,
        include_djvu=args.include_djvu,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
