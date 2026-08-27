from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, features

from .text import normalize_label, validate_assamese_label

_SENTENCE_BOUNDARY = re.compile(r"(?<=[।!?])\s+|\n+")


def stable_split(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def extract_lines(text: str, min_chars: int = 8, max_chars: int = 120) -> Iterator[str]:
    text = normalize_label(text)
    for sentence in _SENTENCE_BOUNDARY.split(text):
        words = sentence.split()
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and len(candidate) > max_chars:
                line = " ".join(current)
                valid, _ = validate_assamese_label(line, min_script_ratio=0.75)
                if len(line) >= min_chars and valid:
                    yield line
                current = [word]
            else:
                current.append(word)
        line = " ".join(current)
        valid, _ = validate_assamese_label(line, min_script_ratio=0.75)
        if min_chars <= len(line) <= max_chars and valid:
            yield line


def prepare_wikipedia_corpus(
    output_path: str | Path, max_lines: int = 250_000
) -> dict[str, object]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repo_id = "wikimedia/wikipedia"
    config_name = "20231101.as"
    revision = HfApi().dataset_info(repo_id).sha
    dataset = load_dataset(repo_id, config_name, split="train", revision=revision)
    counts = {"train": 0, "validation": 0, "test": 0}
    with output_path.open("w", encoding="utf-8") as handle:
        for article in dataset:
            source_id = str(article.get("id") or article.get("title"))
            split = stable_split(source_id)
            for index, line in enumerate(extract_lines(article["text"])):
                if sum(counts.values()) >= max_lines:
                    break
                record = {
                    "id": f"wikipedia:{source_id}:{index}",
                    "text": line,
                    "split": split,
                    "source_language": "as",
                    "source_title": article.get("title"),
                    "license": "CC-BY-SA-3.0/GFDL",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
            if sum(counts.values()) >= max_lines:
                break
    corpus_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata: dict[str, object] = {
        "dataset": repo_id,
        "config": config_name,
        "source_language": "as",
        "revision": revision,
        "license": "CC-BY-SA-3.0/GFDL",
        "counts": counts,
        "sha256": corpus_sha256,
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not features.check_feature("raqm"):
        raise RuntimeError(
            "Pillow was built without libraqm. Complex Assamese shaping is required; "
            "run synthetic generation in the Linux GPU notebook or install a RAQM-enabled Pillow."
        )
    return ImageFont.truetype(str(path), size=size, layout_engine=ImageFont.Layout.RAQM)


def render_assamese_line(text: str, font_path: str | Path, seed: int) -> Image.Image:
    """Render one Assamese line deterministically, including mild scan variation."""
    rng = random.Random(seed)
    text = normalize_label(text)
    font_size = rng.randint(28, 54)
    font = _font(Path(font_path), font_size)
    probe = Image.new("L", (16, 16), 255)
    draw = ImageDraw.Draw(probe)
    try:
        bbox = draw.textbbox((0, 0), text, font=font, language="as", features=["locl"])
    except TypeError:
        bbox = draw.textbbox((0, 0), text, font=font)
    margin_x, margin_y = rng.randint(8, 24), rng.randint(5, 14)
    width = int(max(8, bbox[2] - bbox[0] + margin_x * 2))
    height = int(max(8, bbox[3] - bbox[1] + margin_y * 2))
    background = rng.randint(235, 255)
    foreground = rng.randint(0, 45)
    image = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(image)
    position = (margin_x - bbox[0], margin_y - bbox[1])
    try:
        draw.text(
            position,
            text,
            font=font,
            fill=foreground,
            language="as",
            features=["locl"],
        )
    except TypeError:
        draw.text(position, text, font=font, fill=foreground)
    if rng.random() < 0.45:
        image = image.rotate(rng.uniform(-1.2, 1.2), resample=Image.Resampling.BICUBIC, fillcolor=background)
    if rng.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.4))
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.8)))
    return image


def render_synthetic_dataset(
    corpus_path: str | Path,
    fonts_dir: str | Path,
    output_dir: str | Path,
    samples: int,
    seed: int = 1337,
    overwrite: bool = False,
) -> None:
    from datasets import Dataset, DatasetDict, Features, Value
    from datasets import Image as HFImage

    corpus_path, fonts_dir, output_dir = Path(corpus_path), Path(fonts_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass overwrite=True only if it may be replaced"
            )
        resolved = output_dir.resolve()
        if resolved == Path.cwd().resolve() or resolved.parent == resolved:
            raise ValueError(f"Refusing unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
    records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    train_records = [record for record in records if record["split"] == "train"]
    fonts = sorted([*fonts_dir.glob("*.ttf"), *fonts_dir.glob("*.otf")])
    if not train_records:
        raise ValueError("No Assamese training lines found in the corpus")
    if not fonts:
        raise ValueError(f"No fonts found in {fonts_dir}; run scripts/download_fonts.py")

    def examples() -> Iterator[dict[str, object]]:
        rng = random.Random(seed)
        for index in range(samples):
            record = train_records[rng.randrange(len(train_records))]
            font_path = fonts[rng.randrange(len(fonts))]
            yield {
                "id": f"synthetic:{index:09d}",
                "image": render_assamese_line(record["text"], font_path, seed + index),
                "text": record["text"],
                "source_language": "as",
                "source_config": "20231101.as",
                "font": font_path.name,
            }

    features = Features(
        {
            "id": Value("string"),
            "image": HFImage(),
            "text": Value("string"),
            "source_language": Value("string"),
            "source_config": Value("string"),
            "font": Value("string"),
        }
    )
    dataset = Dataset.from_generator(examples, features=features)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": dataset}).save_to_disk(str(output_dir))
    font_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in fonts
    }
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    (output_dir / "generation.json").write_text(
        json.dumps(
            {
                "language": "as",
                "source_config": "20231101.as",
                "samples": samples,
                "seed": seed,
                "corpus_sha256": corpus_hash,
                "font_sha256": font_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
