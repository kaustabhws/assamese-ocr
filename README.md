# Axomiya Layout OCR

An Assamese-first OCR system for scanned pages that preserves semantic layout, text coordinates, confidence, and reading order. The intended deployment target is CPU inference through ONNX Runtime; GPU is required only for training.

## What this project builds

The system is deliberately modular:

1. **Page layout** identifies titles, paragraphs, lists, tables, figures, headers, footers, forms, and other regions.
2. **Text detection** finds line polygons using a small, script-independent detector.
3. **Assamese recognition** reads each crop with an Assamese-only CRNN-CTC model.
4. **Reconstruction** emits canonical JSON and rebuilds searchable HTML/PDF while retaining positions and reading order.

Assamese and Bengali share most Unicode code points and much of the Eastern Nagari script. This repository does **not** train on the Bengali configuration or a Bengali text corpus. It downloads only Mozhi's `assamese` split and Assamese Wikipedia (`20231101.as`) for optional synthetic lines. Assamese-specific characters such as `ৰ` and `ৱ` receive explicit coverage checks.

## Current status

This repository contains the complete data, training, evaluation, export, and inference code. The ready-to-run INT8 release models are included in `dist/axomiya-ocr-0.2.0`; training checkpoints and generated datasets remain excluded.

## Architecture profiles

| Stage | Balanced profile | Mobile target |
|---|---|---|
| Layout | Docling Heron INT8 ONNX, 17 classes | Fine-tuned SSDLite-MobileNetV3 INT8 |
| Text detection | RapidOCR/Paddle DBNet mobile detector | Same |
| Recognition | Assamese CRNN-CTC, dynamic-width ONNX | INT8 version of same model |
| Output | JSON, HTML overlay, searchable PDF | JSON/HTML first; PDF where supported |

The balanced layout model is about 70 MB. The custom recognizer is designed to stay in the single-digit to low-double-digit MB range after quantization, but its final size and accuracy must be measured after training.

The production recognizer architecture has 3,144,646 parameters. An untrained export smoke test measured 12.58 MB FP32 and 3.68 MB INT8, with 4.07 ms median CPU execution for a `1x1x48x384` tensor on this machine. Those measurements validate the size/latency design only; they are not an accuracy result. The balanced three-model bundle is roughly 83 MB before runtime libraries, dominated by semantic layout.

## Quick start

Use Python 3.10–3.12. For local preparation on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[train,inference,notebooks,dev]"
.\.venv\Scripts\python scripts\validate_environment.py
.\.venv\Scripts\python scripts\prepare_mozhi.py --output data/processed/mozhi_assamese_smoke --max-per-split 64
.\.venv\Scripts\python -m pytest
```

For the complete dataset, omit `--max-per-split`. Then open the notebooks in order:

1. `notebooks/00_environment.ipynb`
2. `notebooks/01_prepare_assamese_data.ipynb`
3. `notebooks/02_train_recognizer.ipynb`
4. `notebooks/03_evaluate_export.ipynb`
5. `notebooks/04_layout_and_end_to_end.ipynb`
6. `notebooks/05_page_benchmark.ipynb`

Each notebook can run on Colab/Kaggle or another CUDA machine after the repository and prepared data are available.

## Run OCR on an image

The repository includes the recognizer, page-layout model, and text-line detector. A GPU is not required for inference.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[inference]"
.\.venv\Scripts\python ocr.py "C:\path\to\assamese-page.jpg"
```

Linux/macOS:

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[inference]"
./.venv/bin/python ocr.py /path/to/assamese-page.jpg
```

By default the results are written to `ocr_output/<input-name>/`:

- `text.txt`: plain UTF-8 Assamese text in reading order
- `document.json`: text, confidence, coordinates, regions, and reading order
- `document.html`: the original page with a selectable OCR text overlay

Choose another output directory with `--output results/my-page`. PDF input is also supported. To additionally create a searchable PDF, pass an Assamese font with `--font path/to/font.ttf`.

## Release gates

A model is not considered ready merely because training finishes. The release report must include:

- Assamese held-out character error rate (CER), word error rate (WER), and exact sequence accuracy
- separate `ৰ`/`ৱ` recall and a list of unseen characters
- clean-scan and degraded-scan results
- end-to-end page CER with detected—not ground-truth—text boxes
- reading-order and layout-region metrics
- FP32 and INT8 accuracy deltas
- model sizes and cold/warm CPU latency on named hardware
- visual review of real Assamese book/newspaper pages

See [docs/TRAINING.md](docs/TRAINING.md) and [docs/DATA.md](docs/DATA.md) for the exact workflow.

The measured baseline is recorded in [docs/BASELINE.md](docs/BASELINE.md). Official Tesseract `asm-fast` reaches 23.17% CER and 54.37% exact word accuracy on the complete cleaned Mozhi test set, giving the trained model a concrete comparison target.

For real-scan improvement, the repository can build a document-disjoint Assamese Wikisource dataset from human-validated page transcriptions and warm-start the current checkpoint with `configs/recognizer_finetune.yaml`. See [docs/DATA.md](docs/DATA.md) and [docs/TRAINING.md](docs/TRAINING.md). The SEBA textbook pages used for regression testing are deliberately excluded from training.

## Data and model licensing

Project code is Apache-2.0. External datasets, fonts, and model weights retain their own licenses. The Mozhi mirror does not state a self-contained redistribution license and points users to the original CVIT source; verify its terms before redistribution or commercial use. The downloader records source revisions so experiments remain auditable.
