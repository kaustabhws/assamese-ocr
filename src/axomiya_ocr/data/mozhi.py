from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .text import (
    ASSAMESE_CONFIG,
    assert_assamese_source,
    audit_texts,
    ctc_required_steps,
    normalize_label,
    validate_assamese_label,
)
from .vocab import Vocabulary

MOZHI_REPO_ID = "darknight054/indic-mozhi-ocr"


def _image_sha256(image: Image.Image) -> str:
    normalized = image.convert("L")
    digest = hashlib.sha256()
    digest.update(f"{normalized.width}x{normalized.height}:L".encode())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    text = normalize_label(example["text"])
    valid, reason = validate_assamese_label(text)
    return {
        "text": text,
        "valid": valid,
        "validation_reason": reason,
        "image_sha256": _image_sha256(example["image"]),
        "image_width": example["image"].width,
        "image_height": example["image"].height,
        "ctc_required_steps": ctc_required_steps(text),
        "source_language": "as",
        "source_config": ASSAMESE_CONFIG,
    }


def _overlap_count(left: set[str], right: set[str]) -> int:
    return len(left.intersection(right))


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def prepare_mozhi(
    output_dir: str | Path,
    *,
    max_per_split: int | None = None,
    num_proc: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download, validate, audit, and save only the Assamese Mozhi config."""
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import HfApi

    assert_assamese_source(ASSAMESE_CONFIG)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass overwrite=True only if it may be replaced"
            )
        resolved = output_dir.resolve()
        if resolved == Path.cwd().resolve() or resolved.parent == resolved:
            raise ValueError(f"Refusing unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
    info = HfApi().dataset_info(MOZHI_REPO_ID)
    revision = info.sha
    raw = load_dataset(MOZHI_REPO_ID, ASSAMESE_CONFIG, revision=revision)

    processed: dict[str, Any] = {}
    excluded: dict[str, dict[str, int]] = {}
    ids: dict[str, set[str]] = {}
    image_hashes: dict[str, set[str]] = {}
    for split_name in ("train", "validation", "test"):
        split = raw[split_name]
        if max_per_split and max_per_split > 0:
            split = split.select(range(min(max_per_split, len(split))))
        split = split.map(
            _normalize_example,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc=f"Validate {split_name}",
        )
        reasons = Counter(split["validation_reason"])
        excluded[split_name] = {
            reason: count for reason, count in reasons.items() if reason != "ok"
        }
        split = split.filter(lambda example: example["valid"], num_proc=num_proc)
        processed[split_name] = split
        ids[split_name] = set(split["id"])
        image_hashes[split_name] = set(split["image_sha256"])

    vocab = Vocabulary.build(processed["train"]["text"])
    unseen: dict[str, list[str]] = {}
    train_chars = set(vocab.characters)
    for split_name in ("validation", "test"):
        split_chars = {char for text in processed[split_name]["text"] for char in text}
        unseen[split_name] = sorted(split_chars - train_chars, key=ord)

    leakage_before = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        leakage_before[f"{left}_{right}"] = {
            "shared_ids": _overlap_count(ids[left], ids[right]),
            "shared_image_hashes": _overlap_count(image_hashes[left], image_hashes[right]),
        }

    valid_sizes = {name: len(split) for name, split in processed.items()}
    train_hashes = set(processed["train"]["image_sha256"])
    processed["validation"] = processed["validation"].filter(
        lambda image_hash: image_hash not in train_hashes,
        input_columns=["image_sha256"],
    )
    evaluation_hashes = train_hashes.union(processed["validation"]["image_sha256"])
    processed["test"] = processed["test"].filter(
        lambda image_hash: image_hash not in evaluation_hashes,
        input_columns=["image_sha256"],
    )
    final_hashes = {
        split_name: set(split["image_sha256"]) for split_name, split in processed.items()
    }
    leakage_after = {
        f"{left}_{right}": _overlap_count(final_hashes[left], final_hashes[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }

    audits: dict[str, Any] = {}
    for split_name, split in processed.items():
        audit = audit_texts(split["text"])
        resized_widths = [
            round(width * 48 / max(1, height))
            for width, height in zip(split["image_width"], split["image_height"], strict=True)
        ]
        audits[split_name] = {
            **audit.to_dict(),
            "excluded_by_reason": excluded[split_name],
            "removed_duplicate_images": valid_sizes[split_name] - len(split),
            "after_filter_and_deduplication": len(split),
            "max_label_codepoints": max(map(len, split["text"]), default=0),
            "max_ctc_steps": max(split["ctc_required_steps"], default=0),
            "resized_width_at_height_48": {
                "median": _percentile(resized_widths, 0.5),
                "p95": _percentile(resized_widths, 0.95),
                "maximum": max(resized_widths, default=0),
                "over_768": sum(width > 768 for width in resized_widths),
            },
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict = DatasetDict(processed)
    dataset_dict.save_to_disk(str(output_dir))
    vocab_path = output_dir / "vocab.json"
    vocab.save(vocab_path)
    report = {
        "dataset": MOZHI_REPO_ID,
        "config": ASSAMESE_CONFIG,
        "source_language": "as",
        "revision": revision,
        "license_note": "Dataset card refers users to the original CVIT source for terms.",
        "max_per_split": max_per_split,
        "vocab_size_including_blank": vocab.size,
        "vocab_sha256": vocab.sha256,
        "unseen_characters": unseen,
        "split_leakage_before_cleanup": leakage_before,
        "split_leakage_after_cleanup": leakage_after,
        "splits": audits,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
