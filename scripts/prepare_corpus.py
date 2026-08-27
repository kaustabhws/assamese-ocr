from __future__ import annotations

import argparse
import json

from axomiya_ocr.data.synthetic import prepare_wikipedia_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Assamese-only synthetic line text")
    parser.add_argument("--output", default="data/processed/assamese_corpus.jsonl")
    parser.add_argument("--max-lines", type=int, default=250_000)
    args = parser.parse_args()
    report = prepare_wikipedia_corpus(args.output, max_lines=args.max_lines)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
