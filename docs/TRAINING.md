# GPU training runbook

## Recommended machine

- NVIDIA GPU with at least 12 GB VRAM for the recognizer
- 24 GB or more for comfortable layout-detector fine-tuning
- 4+ CPU workers and 20 GB free storage
- Python 3.10–3.12 with a CUDA-compatible PyTorch build

## Reproducible sequence

```bash
python -m pip install -U pip
python -m pip install -e ".[train,inference,notebooks,dev]"
python scripts/prepare_mozhi.py
python scripts/download_fonts.py
python scripts/prepare_corpus.py --max-lines 250000
python scripts/render_synthetic.py --samples 250000
python scripts/download_tesseract_baselines.py
python scripts/evaluate_tesseract.py --profile fast --split test
python scripts/train_recognizer.py --config configs/recognizer.yaml
python scripts/evaluate_recognizer.py \
  --checkpoint artifacts/recognizer/best.pt \
  --dataset data/processed/mozhi_assamese \
  --split test
python scripts/export_recognizer.py \
  --checkpoint artifacts/recognizer/best.pt \
  --output artifacts/recognizer/assamese_recognizer.onnx \
  --quantize
python scripts/benchmark_onnx.py \
  --model artifacts/recognizer/assamese_recognizer.int8.onnx
python scripts/check_release.py
python scripts/build_bundle.py
```

The notebooks call the same scripts, so notebook and command-line runs use identical implementation code.

If a GPU session stops, resume without resetting the optimizer or scheduler:

```bash
python scripts/train_recognizer.py --config configs/recognizer.yaml \
  --resume artifacts/recognizer/last.pt
```

## Fine-tune the existing recognizer on real scanned lines

Prepare the validated Wikisource data first using the command in `docs/DATA.md`. Then warm-start from the existing best checkpoint:

```bash
python scripts/train_recognizer.py \
  --config configs/recognizer_finetune.yaml \
  --init-from artifacts/recognizer/best.pt
```

`--init-from` loads model weights only and starts a new low-learning-rate run in `artifacts/recognizer_finetuned`. This is different from `--resume`, which restores the optimizer, scheduler, epoch and random state after an interrupted run. Never use both options together.

The fine-tune profile uses 1024-pixel line inputs, 10% synthetic replay to reduce forgetting, stronger real-scan augmentation, and 2x sampling for labels containing the character families implicated by the SEBA review. It keeps the exact original vocabulary and architecture, so the existing checkpoint remains compatible.

After training, evaluate both the new real-scan split and the untouched Mozhi test split:

```bash
python scripts/evaluate_recognizer.py \
  --checkpoint artifacts/recognizer_finetuned/best.pt \
  --dataset data/processed/wikisource_assamese \
  --split test \
  --output artifacts/recognizer_finetuned/wikisource_test_metrics.json

python scripts/evaluate_recognizer.py \
  --checkpoint artifacts/recognizer_finetuned/best.pt \
  --dataset data/processed/mozhi_assamese \
  --split test \
  --output artifacts/recognizer_finetuned/mozhi_test_metrics.json
```

Export only after both evaluations. Re-run the frozen SEBA 20-page regression set and compare it visually before replacing the old model.

```bash
python scripts/export_recognizer.py \
  --checkpoint artifacts/recognizer_finetuned/best.pt \
  --output artifacts/recognizer_finetuned/assamese_recognizer.onnx \
  --quantize
```

## Training decisions

- Primary checkpoint metric: validation CER.
- Stop early after seven epochs without a CER improvement.
- Save optimizer, scheduler, scaler, vocabulary hash, config, and source revisions in every checkpoint.
- Use mixed real-word and synthetic-line batches; start with 35% synthetic samples and tune only on validation.
- Do not select a checkpoint using the test split.
- Export FP32 first, verify numerical parity, then quantize and measure the CER delta.

## Acceptance targets

Targets are deliberately not presented as achieved results:

- Mozhi test CER <= 3% for the first release candidate
- `ৰ` and `ৱ` recall >= 97% where each occurs at least 100 times
- INT8 CER regression <= 0.3 percentage points
- recognizer INT8 size <= 15 MB
- recognizer median CPU latency <= 25 ms per typical line on a named modern laptop CPU
- zero unknown characters on the approved real-page benchmark

If these gates fail, preserve the benchmark and adjust training data/model capacity rather than weakening the test.
