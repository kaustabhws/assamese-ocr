# Data protocol

## Recognition sources

### Real printed words

- Dataset: Mozhi printed OCR dataset
- Mirror: `darknight054/indic-mozhi-ocr`
- Configuration: **only** `assamese`
- Official counts in the mirror: 79,697 train; 9,945 validation; 10,146 test
- Original project: <https://cvit.iiit.ac.in/usodi/tdocrmil.php>
- Paper: Mathew, Mondal, and Jawahar, *Towards Deployable OCR Models for Indic Languages*

The source card says to consult the original project for licensing. Do not redistribute the pixels or use them commercially until the terms are confirmed.

The local audit at source revision `521143f1549bc6e5ef9125af75f119740e860171` retained 79,655 train, 9,941 validation, and 10,140 test samples. It excluded 43 labels whose letters were not predominantly from the Assamese/Eastern Nagari block and removed nine evaluation images duplicated across official splits. Cross-split exact-image overlap is zero after cleanup. The machine-readable evidence is in `data/processed/mozhi_assamese/audit.json` after preparation.

### Synthetic Assamese lines

Text comes from Assamese Wikipedia config `20231101.as` (CC BY-SA 3.0 and GFDL per its dataset card), or from a user-owned UTF-8 corpus. Rendering uses shared Eastern Nagari fonts with Assamese language shaping (`language="as"`) and the OpenType `locl` feature where Pillow/RAQM supports it. No Bengali-language text is added.

The prepared corpus pins revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` and contains 221,940 train, 14,190 validation, and 13,870 test lines. Only the training lines are rendered for augmentation; validation and test remain available for corpus auditing and are not used to tune OCR weights.

Synthetic data is useful for fonts, long lines, punctuation, and degradation. It is not a replacement for a manually verified real-page test set.

### Validated Assamese Wikisource scans

`scripts/prepare_wikisource.py` creates real scanned-line training data from the Assamese Wikisource `Page:` namespace. It accepts only pages at ProofreadPage quality level 4 (validated), records the exact page revision, verifies that each source scan has Public Domain or Creative Commons metadata, and records file-level attribution. Assamese Wikisource transcriptions are CC BY-SA 4.0.

By default it selects PDF-backed books, downloads each source file once, and renders selected pages locally at 200 DPI. This avoids low-resolution thumbnails and Wikimedia request throttling. `--include-djvu` broadens typeface coverage but is slower because DjVu pages are fetched individually. API responses and source files are cached under `data/raw/wikisource_cache`, so interrupted preparation can be rerun safely.

Whole source books are assigned to train, validation, or test before pages are selected, preventing crops from one book leaking across splits. The current recognizer is used only to align detected image lines to the human transcription. The saved label is the human transcription, never the bootstrap prediction. Lines with weak alignment, unknown vocabulary characters, impossible CTC lengths, excessive non-Assamese text, or duplicate crop pixels are rejected.

This is intentionally described as **silver line data**: the page transcription is human validated, but line segmentation and alignment are automatic. Use it to fine-tune the recognizer; retain a separately hand-reviewed page benchmark for final accuracy claims. Dataset files are downloaded/generated locally and are not committed.

Recommended preparation:

```bash
python scripts/prepare_wikisource.py \
  --recognizer artifacts/recognizer/assamese_recognizer.int8.onnx \
  --vocab data/processed/mozhi_assamese/vocab.json \
  --train-documents 12 \
  --validation-documents 2 \
  --test-documents 2 \
  --pages-per-document 30 \
  --output data/processed/wikisource_assamese
```

Review `data/processed/wikisource_assamese/audit.json`, `page_audit.jsonl`, and `attribution.json` before training. Do not use `--allow-unknown-license` unless you have independently established the missing scan rights.

Record the current model's baseline on the accepted aligned lines before fine-tuning:

```bash
python scripts/evaluate_bootstrap.py \
  --dataset data/processed/wikisource_assamese \
  --output artifacts/wikisource_bootstrap_metrics.json
```

## Preparation rules

1. Preserve the official Mozhi train/validation/test assignment; remove exact pixel duplicates from validation/test rather than moving examples between splits.
2. Normalize labels with Unicode NFC and collapse spacing; do not rewrite characters.
3. Reject empty labels, control characters, excessive non-Assamese letters, corrupt images, and impossible CTC alignments.
4. Build the vocabulary from the training split plus explicit Assamese-required characters. Report—but never silently absorb—unknown validation/test symbols.
5. Hash source image bytes and normalized text to find exact cross-split leakage.
6. For user page data, split by book/document before creating crops. Pages from one book must not leak across train and test.
7. Keep the official test split untouched. Tune thresholds only on validation.
8. Never train on the 20 SEBA pages used for the current real-page check; they remain a regression benchmark.

## Required real-page benchmark

Before calling the project accurate, create a small gold set covering at least:

- modern clean print
- older typefaces
- skewed and photographed pages
- two-column newspaper pages
- tables/forms
- pages containing `ৰ` and `ৱ`

Recommended minimum: 100 pages from at least 10 source documents, with document-level splits and independently checked transcriptions. The repository includes the canonical annotation schema used by the reconstruction code.
