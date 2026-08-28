from __future__ import annotations

import hashlib
import html
import io
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from PIL import Image

from axomiya_ocr.inference.recognizer import ONNXRecognizer
from axomiya_ocr.inference.text_detector import RapidTextDetector, rectify_crop
from axomiya_ocr.training.metrics import edit_distance

from .text import (
    audit_texts,
    ctc_required_steps,
    normalize_label,
    validate_assamese_label,
)
from .vocab import Vocabulary

WIKISOURCE_API = "https://as.wikisource.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "AxomiyaLayoutOCR/0.1 "
    "(https://github.com/kaustabhws/assamese-ocr; Assamese OCR dataset preparation)"
)
TRANSCRIPTION_LICENSE = "CC BY-SA 4.0"

_BLOCK_TAGS = {"br", "div", "h1", "h2", "h3", "h4", "li", "p", "table", "tr"}
_THUMB_QUERY = re.compile(r"\?.*$")


@dataclass(frozen=True)
class DetectedLine:
    polygon: list[list[float]]
    crop: Image.Image
    prediction: str
    confidence: float


@dataclass(frozen=True)
class AlignedLine:
    index: int
    prediction: str
    transcription: str
    alignment_score: float


class _PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._target_depth = 0
        self._seen_target = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "") or ""
        if not self._target_depth and "pagetext" in classes.split():
            self._target_depth = 1
            self._seen_target = True
            return
        if self._target_depth:
            self._target_depth += 1
            if tag in _BLOCK_TAGS:
                self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._target_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._target_depth:
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        self._target_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._target_depth:
            self.parts.append(data)

    @property
    def text(self) -> str:
        if not self._seen_target:
            return ""
        return normalize_label(html.unescape("".join(self.parts)).replace("\u00ad", ""))


class MediaWikiClient:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None = "data/raw/wikisource_cache",
        pause_seconds: float = 0.25,
        retries: int = 6,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pause_seconds = pause_seconds
        self.retries = retries

    def _cache_path(self, group: str, key: str, suffix: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(key.encode()).hexdigest()
        path = self.cache_dir / group / f"{digest}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _request(self, endpoint: str, params: dict[str, Any], *, post: bool = False) -> Any:
        request_params = {
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
            **{key: str(value) for key, value in params.items()},
        }
        cache_path = self._cache_path(
            "api",
            endpoint + "\n" + json.dumps(request_params, ensure_ascii=False, sort_keys=True),
            "json",
        )
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        encoded = urllib.parse.urlencode(request_params).encode("utf-8")
        request = urllib.request.Request(
            endpoint if post else f"{endpoint}?{encoded.decode('ascii')}",
            data=encoded if post else None,
            headers={"User-Agent": USER_AGENT},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                if "error" in payload:
                    raise RuntimeError(f"MediaWiki API error: {payload['error']}")
                if cache_path:
                    cache_path.write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )
                time.sleep(self.pause_seconds)
                return payload
            except HTTPError as error:  # pragma: no cover - network retry path
                last_error = error
                if error.code != 429:
                    time.sleep(min(8.0, 0.5 * (2**attempt)))
                    continue
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 5.0 * (attempt + 1))
                time.sleep(delay)
            except Exception as error:  # pragma: no cover - network retry path
                last_error = error
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"MediaWiki request failed after {self.retries} attempts") from last_error

    def index_titles(self) -> list[str]:
        titles: list[str] = []
        continuation: dict[str, Any] = {}
        while True:
            payload = self._request(
                WIKISOURCE_API,
                {
                    "action": "query",
                    "list": "allpages",
                    "apnamespace": 106,
                    "aplimit": "max",
                    **continuation,
                },
            )
            titles.extend(page["title"] for page in payload["query"]["allpages"])
            continuation = payload.get("continue", {})
            if not continuation:
                return titles

    def pages_in_index(self, index_title: str) -> list[dict[str, Any]]:
        payload = self._request(
            WIKISOURCE_API,
            {
                "action": "query",
                "list": "proofreadpagesinindex",
                "prppiititle": index_title,
                "prppiiprop": "ids|title|formattedpagenumber",
            },
        )
        return list(payload.get("query", {}).get("proofreadpagesinindex", []))

    def page_details(self, titles: Sequence[str]) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        for start in range(0, len(titles), 25):
            payload = self._request(
                WIKISOURCE_API,
                {
                    "action": "query",
                    "prop": "proofread|imageforpage|revisions",
                    "prppifpprop": "filename|size|fullsize",
                    "rvprop": "ids|timestamp",
                    "titles": "|".join(titles[start : start + 25]),
                },
                post=True,
            )
            pages.extend(payload.get("query", {}).get("pages", []))
        return pages

    def parsed_page_text(self, revision_id: int) -> str:
        payload = self._request(
            WIKISOURCE_API,
            {
                "action": "parse",
                "oldid": revision_id,
                "prop": "text",
                "disableeditsection": 1,
                "disablelimitreport": 1,
            },
            post=True,
        )
        parser = _PageTextParser()
        parser.feed(payload.get("parse", {}).get("text", ""))
        return parser.text

    def file_metadata(self, filename: str) -> dict[str, str]:
        payload = self._request(
            COMMONS_API,
            {
                "action": "query",
                "prop": "imageinfo",
                "iiprop": "url|sha1|extmetadata",
                "titles": f"File:{filename}",
            },
            post=True,
        )
        page = payload.get("query", {}).get("pages", [{}])[0]
        image_info = page.get("imageinfo", [{}])[0]
        metadata = image_info.get("extmetadata", {})

        def value(name: str) -> str:
            return str(metadata.get(name, {}).get("value", ""))

        return {
            "filename": filename,
            "file_url": _canonical_url(str(image_info.get("url", ""))),
            "sha1": str(image_info.get("sha1", "")),
            "license": value("LicenseShortName"),
            "license_url": value("LicenseUrl"),
            "artist": _strip_html(value("Artist")),
            "credit": _strip_html(value("Credit")),
        }

    def download(self, url: str) -> bytes:
        url = _canonical_url(url)
        cache_path = self._cache_path("images", url, "bin")
        if cache_path and cache_path.exists():
            return cache_path.read_bytes()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    content = response.read()
                if cache_path:
                    cache_path.write_bytes(content)
                time.sleep(self.pause_seconds)
                return content
            except HTTPError as error:  # pragma: no cover - network retry path
                last_error = error
                if error.code != 429:
                    time.sleep(min(8.0, 0.5 * (2**attempt)))
                    continue
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 5.0 * (attempt + 1))
                time.sleep(delay)
            except Exception as error:  # pragma: no cover - network retry path
                last_error = error
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"Image download failed after {self.retries} attempts: {url}") from last_error


def _strip_html(value: str) -> str:
    return normalize_label(re.sub(r"<[^>]+>", " ", html.unescape(value)))


def _canonical_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    return _THUMB_QUERY.sub("", url)


def _stable_value(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16)


def document_split(title: str, seed: int = 1337) -> str:
    bucket = _stable_value(title, seed) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def select_documents(
    titles: Iterable[str], counts: dict[str, int], seed: int = 1337
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {split: [] for split in counts}
    ordered = sorted(set(titles), key=lambda title: (_stable_value(title, seed), title))
    for title in ordered:
        split = document_split(title, seed)
        if split in counts and len(selected[split]) < counts[split]:
            selected[split].append(title)
        if all(len(selected[split]) >= count for split, count in counts.items()):
            break
    missing = {split: counts[split] - len(values) for split, values in selected.items() if len(values) < counts[split]}
    if missing:
        raise ValueError(f"Not enough Wikisource indexes for requested splits: {missing}")
    return selected


def page_text_from_html(markup: str) -> str:
    parser = _PageTextParser()
    parser.feed(markup)
    return parser.text


def _boundary_map(source: str, target: str) -> list[int]:
    mapping: list[int | None] = [None] * (len(source) + 1)
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    for _, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        width = source_end - source_start
        if width == 0:
            existing = mapping[source_start]
            mapping[source_start] = max(existing or 0, target_end)
            continue
        for offset in range(width + 1):
            fraction = offset / width
            mapped = round(target_start + fraction * (target_end - target_start))
            mapping[source_start + offset] = mapped
    last = 0
    output: list[int] = []
    for value in mapping:
        if value is None:
            value = last
        value = max(last, min(len(target), value))
        output.append(value)
        last = value
    output[-1] = len(target)
    return output


def align_transcription(predictions: Sequence[str], transcription: str) -> list[AlignedLine]:
    normalized_predictions = [normalize_label(value) for value in predictions]
    transcription = normalize_label(transcription)
    combined_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for prediction in normalized_predictions:
        if combined_parts:
            combined_parts.append(" ")
            cursor += 1
        start = cursor
        combined_parts.append(prediction)
        cursor += len(prediction)
        spans.append((start, cursor))
    combined = "".join(combined_parts)
    if not combined or not transcription:
        return []
    boundaries = _boundary_map(combined, transcription)
    aligned: list[AlignedLine] = []
    for index, ((start, end), prediction) in enumerate(zip(spans, normalized_predictions, strict=True)):
        reference = normalize_label(transcription[boundaries[start] : boundaries[end]])
        denominator = max(1, len(reference), len(prediction))
        score = max(0.0, 1.0 - edit_distance(reference, prediction) / denominator)
        aligned.append(AlignedLine(index, prediction, reference, score))
    return aligned


def _line_box(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _order_polygons(
    polygons: Sequence[list[list[float]]], page_width: int
) -> list[list[list[float]]]:
    boxes = [(polygon, _line_box(polygon)) for polygon in polygons]
    full_width = [item for item in boxes if (item[1][2] - item[1][0]) / max(1, page_width) >= 0.62]
    narrow = [item for item in boxes if item not in full_width]
    if len(narrow) < 6:
        return [item[0] for item in sorted(boxes, key=lambda item: (item[1][1], item[1][0]))]

    centers = sorted((box[0] + box[2]) / 2 for _, box in narrow)
    gaps = [(right - left, left, right) for left, right in zip(centers, centers[1:], strict=False)]
    gap, left, right = max(gaps, default=(0.0, 0.0, 0.0))
    split_x = (left + right) / 2
    left_column = [item for item in narrow if (item[1][0] + item[1][2]) / 2 < split_x]
    right_column = [item for item in narrow if item not in left_column]
    has_columns = (
        gap >= page_width * 0.12
        and len(left_column) >= 3
        and len(right_column) >= 3
        and max(box[2] for _, box in left_column) < min(box[0] for _, box in right_column)
    )
    if not has_columns:
        return [item[0] for item in sorted(boxes, key=lambda item: (item[1][1], item[1][0]))]

    first_column_y = min(box[1] for _, box in narrow)
    prefix = [item for item in full_width if item[1][1] < first_column_y]
    suffix = [item for item in full_width if item not in prefix]
    ordered = [
        *sorted(prefix, key=lambda item: (item[1][1], item[1][0])),
        *sorted(left_column, key=lambda item: (item[1][1], item[1][0])),
        *sorted(right_column, key=lambda item: (item[1][1], item[1][0])),
        *sorted(suffix, key=lambda item: (item[1][1], item[1][0])),
    ]
    return [item[0] for item in ordered]


def _detect_lines(
    image: Image.Image, detector: RapidTextDetector, recognizer: ONNXRecognizer
) -> list[DetectedLine]:
    polygons = _order_polygons(detector.predict(image), image.width)
    lines: list[DetectedLine] = []
    for polygon in polygons:
        crop = rectify_crop(image, polygon)
        prediction, confidence = recognizer.predict(crop)
        prediction = normalize_label(prediction)
        if prediction:
            lines.append(DetectedLine(polygon, crop, prediction, confidence))
    return lines


def _license_is_usable(license_name: str) -> bool:
    normalized = license_name.casefold()
    return "public domain" in normalized or normalized.startswith("cc ") or normalized == "cc0"


def _safe_replace_directory(path: Path, overwrite: bool) -> None:
    if not path.exists() or not any(path.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    resolved = path.resolve()
    if resolved == Path.cwd().resolve() or resolved.parent == resolved:
        raise ValueError(f"Refusing unsafe output directory: {resolved}")
    shutil.rmtree(resolved)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _render_pdf_page(document_bytes: bytes, page_offset: int, dpi: int = 200) -> Image.Image:
    import fitz

    with fitz.open(stream=document_bytes, filetype="pdf") as document:
        page_index = page_offset - 1
        if not 0 <= page_index < document.page_count:
            raise IndexError(
                f"Page offset {page_offset} is outside a {document.page_count}-page PDF"
            )
        pixmap = document[page_index].get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def prepare_wikisource(
    output_dir: str | Path,
    *,
    recognizer_path: str | Path,
    metadata_path: str | Path | None,
    vocab_path: str | Path,
    train_documents: int = 12,
    validation_documents: int = 2,
    test_documents: int = 2,
    pages_per_document: int = 30,
    min_alignment_score: float = 0.68,
    max_image_width: int = 1024,
    image_height: int = 48,
    seed: int = 1337,
    overwrite: bool = False,
    allow_unknown_license: bool = False,
    cache_dir: str | Path | None = "data/raw/wikisource_cache",
    include_djvu: bool = False,
) -> dict[str, Any]:
    """Build real Assamese line crops from validated Wikisource page transcriptions."""
    from datasets import Dataset, DatasetDict, Features, Value
    from datasets import Image as HFImage
    from tqdm.auto import tqdm

    if not 0.0 <= min_alignment_score <= 1.0:
        raise ValueError("min_alignment_score must be in [0, 1]")
    if pages_per_document < 1:
        raise ValueError("pages_per_document must be positive")
    output_dir = Path(output_dir)
    _safe_replace_directory(output_dir, overwrite)

    client = MediaWikiClient(cache_dir=cache_dir)
    counts = {
        "train": train_documents,
        "validation": validation_documents,
        "test": test_documents,
    }
    counts = {split: count for split, count in counts.items() if count > 0}
    candidate_counts = {
        split: min(max(count * 12, count + 8), 200) for split, count in counts.items()
    }
    index_titles = [
        title
        for title in client.index_titles()
        if title.casefold().endswith(".pdf")
        or (include_djvu and title.casefold().endswith(".djvu"))
    ]
    candidate_documents = select_documents(index_titles, candidate_counts, seed)
    documents: dict[str, list[str]] = {split: [] for split in counts}
    detector = RapidTextDetector()
    recognizer = ONNXRecognizer(recognizer_path, metadata_path)
    vocabulary = Vocabulary.load(vocab_path)
    if recognizer.vocab.sha256 != vocabulary.sha256:
        raise ValueError("Recognizer metadata and requested training vocabulary differ")
    allowed_characters = set(vocabulary.characters)

    features = Features(
        {
            "id": Value("string"),
            "image": HFImage(),
            "text": Value("string"),
            "source_language": Value("string"),
            "source": Value("string"),
            "source_document": Value("string"),
            "source_page": Value("string"),
            "source_revision": Value("int64"),
            "source_image_sha256": Value("string"),
            "source_license": Value("string"),
            "transcription_license": Value("string"),
            "alignment_score": Value("float32"),
            "bootstrap_prediction": Value("string"),
            "bootstrap_confidence": Value("float32"),
        }
    )
    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in counts}
    rejections: Counter[str] = Counter()
    attribution: dict[str, dict[str, str]] = {}
    used_attribution: set[str] = set()
    seen_crop_hashes: set[str] = set()
    page_audit: list[dict[str, Any]] = []

    for split, index_titles in candidate_documents.items():
        candidates = tqdm(index_titles, desc=f"Wikisource {split} documents")
        for index_title in candidates:
            if len(documents[split]) >= counts[split]:
                break
            index_filename = index_title.split(":", 1)[-1]
            if index_filename not in attribution:
                attribution[index_filename] = client.file_metadata(index_filename)
            index_metadata = attribution[index_filename]
            if not _license_is_usable(index_metadata["license"]) and not allow_unknown_license:
                rejections["document_unusable_or_unknown_scan_license"] += 1
                continue
            records_before_document = len(split_records[split])
            pages = client.pages_in_index(index_title)
            ranked_pages = sorted(
                pages, key=lambda page: (_stable_value(page["title"], seed), page["title"])
            )
            eligible: list[dict[str, Any]] = []
            for batch_start in range(0, len(ranked_pages), 25):
                batch = ranked_pages[batch_start : batch_start + 25]
                details = client.page_details([page["title"] for page in batch])
                details_by_title = {page["title"]: page for page in details}
                for page in batch:
                    detail = details_by_title.get(page["title"], {})
                    if int(detail.get("proofread", {}).get("quality", 0)) != 4:
                        rejections["page_not_validated"] += 1
                        continue
                    image_info = detail.get("imagesforpage")
                    revision = (detail.get("revisions") or [{}])[0]
                    if not image_info or not image_info.get("fullsize") or not revision.get("revid"):
                        rejections["page_missing_image_or_revision"] += 1
                        continue
                    detail["_page_offset"] = int(page["pageoffset"])
                    eligible.append(detail)
                    if len(eligible) >= pages_per_document:
                        break
                if len(eligible) >= pages_per_document:
                    break

            document_bytes = (
                client.download(index_metadata["file_url"])
                if eligible and index_filename.casefold().endswith(".pdf")
                else None
            )

            for detail in eligible:
                image_info = detail["imagesforpage"]
                filename = index_filename if document_bytes is not None else str(
                    image_info.get("filename", "")
                )
                if document_bytes is not None:
                    source_metadata = index_metadata
                else:
                    if filename not in attribution:
                        attribution[filename] = client.file_metadata(filename)
                    source_metadata = attribution[filename]
                if not _license_is_usable(source_metadata["license"]) and not allow_unknown_license:
                    rejections["unusable_or_unknown_scan_license"] += 1
                    continue
                revision = detail["revisions"][0]
                transcription = client.parsed_page_text(int(revision["revid"]))
                valid_page, reason = validate_assamese_label(transcription, min_script_ratio=0.75)
                if len(transcription) < 24:
                    rejections["page_text_too_short"] += 1
                    continue
                if not valid_page:
                    rejections[f"page_text_{reason}"] += 1
                    continue
                if document_bytes is not None:
                    try:
                        image = _render_pdf_page(
                            document_bytes, int(detail["_page_offset"]), dpi=200
                        )
                    except (IndexError, RuntimeError, ValueError):
                        rejections["pdf_page_render_failed"] += 1
                        continue
                    image_bytes = _png_bytes(image)
                else:
                    image_bytes = client.download(str(image_info["fullsize"]))
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                source_image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                detected = _detect_lines(image, detector, recognizer)
                aligned = align_transcription(
                    [line.prediction for line in detected], transcription
                )
                accepted = 0
                for result in aligned:
                    line = detected[result.index]
                    text = result.transcription
                    if result.alignment_score < min_alignment_score:
                        rejections["low_alignment_score"] += 1
                        continue
                    valid, label_reason = validate_assamese_label(text, min_script_ratio=0.75)
                    if len(text) < 3:
                        rejections["line_text_too_short"] += 1
                        continue
                    if not valid:
                        rejections[f"line_text_{label_reason}"] += 1
                        continue
                    if not set(text).issubset(allowed_characters):
                        rejections["outside_checkpoint_vocabulary"] += 1
                        continue
                    if ctc_required_steps(text) * 4 > max_image_width:
                        rejections["ctc_too_long"] += 1
                        continue
                    resized_width = round(line.crop.width * image_height / max(1, line.crop.height))
                    if resized_width > round(max_image_width * 1.15):
                        rejections["crop_too_wide"] += 1
                        continue
                    crop_bytes = _png_bytes(line.crop)
                    crop_hash = hashlib.sha256(crop_bytes).hexdigest()
                    if crop_hash in seen_crop_hashes:
                        rejections["duplicate_crop"] += 1
                        continue
                    seen_crop_hashes.add(crop_hash)
                    box = _line_box(line.polygon)
                    sample_id = hashlib.sha256(
                        f"{detail['title']}:{box}:{text}".encode()
                    ).hexdigest()[:24]
                    split_records[split].append(
                        {
                            "id": f"wikisource:{sample_id}",
                            "image": {"bytes": crop_bytes, "path": None},
                            "text": text,
                            "source_language": "as",
                            "source": "Assamese Wikisource",
                            "source_document": index_title,
                            "source_page": detail["title"],
                            "source_revision": int(revision["revid"]),
                            "source_image_sha256": source_image_sha256,
                            "source_license": source_metadata["license"],
                            "transcription_license": TRANSCRIPTION_LICENSE,
                            "alignment_score": result.alignment_score,
                            "bootstrap_prediction": result.prediction,
                            "bootstrap_confidence": line.confidence,
                        }
                    )
                    accepted += 1
                page_audit.append(
                    {
                        "split": split,
                        "document": index_title,
                        "page": detail["title"],
                        "revision": int(revision["revid"]),
                        "detected_lines": len(detected),
                        "accepted_lines": accepted,
                        "source_image_sha256": source_image_sha256,
                    }
                )
            if len(split_records[split]) > records_before_document:
                documents[split].append(index_title)
                used_attribution.add(filename)
                candidates.set_postfix(
                    documents=f"{len(documents[split])}/{counts[split]}",
                    lines=len(split_records[split]),
                )

    missing_documents = {
        split: counts[split] - len(documents[split])
        for split in counts
        if len(documents[split]) < counts[split]
    }
    if missing_documents:
        raise RuntimeError(
            "Could not find enough usable Wikisource documents: "
            f"{missing_documents}. Rejections: {dict(sorted(rejections.items()))}"
        )

    empty_splits = [split for split, records in split_records.items() if not records]
    if empty_splits:
        raise RuntimeError(
            "No accepted lines were produced for split(s) "
            f"{', '.join(empty_splits)}. Increase documents/pages or inspect filters. "
            f"Rejections: {dict(sorted(rejections.items()))}"
        )
    datasets = {
        split: Dataset.from_list(records, features=features)
        for split, records in split_records.items()
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(datasets).save_to_disk(str(output_dir))
    text_audits = {
        split: audit_texts(record["text"] for record in records).to_dict()
        for split, records in split_records.items()
    }
    report = {
        "dataset": "Assamese Wikisource validated Page namespace",
        "api": WIKISOURCE_API,
        "language": "as",
        "minimum_page_quality": 4,
        "transcription_license": TRANSCRIPTION_LICENSE,
        "seed": seed,
        "documents": documents,
        "pages_per_document": pages_per_document,
        "source_file_types": ["pdf", "djvu"] if include_djvu else ["pdf"],
        "minimum_alignment_score": min_alignment_score,
        "vocabulary_sha256": vocabulary.sha256,
        "splits": {
            split: {"samples": len(records), "text_audit": text_audits[split]}
            for split, records in split_records.items()
        },
        "rejections": dict(sorted(rejections.items())),
        "important_limitation": (
            "Page transcriptions are human validated, but line boundaries are automatically aligned. "
            "Treat this as silver training data, not as the final human-reviewed page benchmark."
        ),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "page_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in page_audit),
        encoding="utf-8",
    )
    (output_dir / "attribution.json").write_text(
        json.dumps(
            [
                metadata
                for key, metadata in attribution.items()
                if key in used_attribution
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
