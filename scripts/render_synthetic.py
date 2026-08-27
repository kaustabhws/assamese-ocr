from __future__ import annotations

import argparse

from axomiya_ocr.data.synthetic import render_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Assamese synthetic OCR lines")
    parser.add_argument("--corpus", default="data/processed/assamese_corpus.jsonl")
    parser.add_argument("--fonts", default="assets/fonts")
    parser.add_argument("--output", default="data/processed/synthetic_assamese")
    parser.add_argument("--vocab", default="data/processed/mozhi_assamese/vocab.json")
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    render_synthetic_dataset(
        args.corpus,
        args.fonts,
        args.output,
        args.samples,
        args.seed,
        overwrite=args.overwrite,
        vocab_path=args.vocab,
    )
    print(f"Saved {args.samples} Assamese synthetic lines to {args.output}")


if __name__ == "__main__":
    main()
