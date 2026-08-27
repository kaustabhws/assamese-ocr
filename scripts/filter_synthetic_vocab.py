from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

from axomiya_ocr.data.text import normalize_label
from axomiya_ocr.data.vocab import Vocabulary


def main() -> None:
    from datasets import DatasetDict, load_from_disk

    parser = argparse.ArgumentParser(
        description="Remove synthetic samples containing characters outside the OCR vocabulary"
    )
    parser.add_argument("--dataset", default="data/processed/synthetic_assamese")
    parser.add_argument("--vocab", default="data/processed/mozhi_assamese/vocab.json")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--promote-existing", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    temporary_path = dataset_path.with_name(f"{dataset_path.name}.filtered")
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_path)

    vocabulary = Vocabulary.load(args.vocab)
    allowed = set(vocabulary.characters)

    if args.promote_existing:
        repaired = load_from_disk(str(temporary_path))
        invalid_count = sum(
            not set(normalize_label(text)).issubset(allowed)
            for text in repaired["train"]["text"]
        )
        if invalid_count:
            raise ValueError(f"Filtered dataset still contains {invalid_count} invalid labels")
        kept_count = len(repaired["train"])
        del repaired
        gc.collect()
        shutil.rmtree(dataset_path)
        shutil.move(str(temporary_path), str(dataset_path))
        print(json.dumps({"kept_samples": kept_count, "invalid_samples": 0}, indent=2))
        return

    if temporary_path.exists():
        raise FileExistsError(f"Remove stale repair directory first: {temporary_path}")

    dataset = load_from_disk(str(dataset_path))
    original_count = len(dataset["train"])

    def is_compatible(text: str) -> bool:
        return set(normalize_label(text)).issubset(allowed)

    filtered_train = dataset["train"].filter(
        is_compatible,
        input_columns=["text"],
        num_proc=args.num_proc,
        desc="Filtering unsupported characters",
    )
    kept_count = len(filtered_train)
    repaired = DatasetDict({**dataset, "train": filtered_train})
    repaired.save_to_disk(str(temporary_path))

    metadata_path = dataset_path / "generation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["samples"] = kept_count
    metadata["vocabulary_sha256"] = vocabulary.sha256
    metadata["vocabulary_rejected_samples"] = original_count - kept_count
    (temporary_path / "generation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    del repaired, filtered_train, dataset
    gc.collect()
    shutil.rmtree(dataset_path)
    shutil.move(str(temporary_path), str(dataset_path))
    print(
        json.dumps(
            {
                "original_samples": original_count,
                "kept_samples": kept_count,
                "removed_samples": original_count - kept_count,
                "vocabulary_sha256": vocabulary.sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
