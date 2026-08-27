# Baseline results

Measured 2026-08-27 on the complete cleaned Mozhi Assamese test split. This is word-crop recognition, not end-to-end page OCR.

## Tesseract `asm-fast`

- Tesseract: 5.5.3
- Assamese traineddata SHA-256: `299c7f6135ac72ca4820d4c39e3cf65b32b24127fc78c4938d11153adbb9fa77`
- Page segmentation mode: 7 (single text line)
- Test samples: 10,140
- Character error rate: **23.17%**
- Word error rate: **47.96%**
- Exact sequence accuracy: **54.37%**
- `ৰ` aligned recall: **92.59%** over 3,265 occurrences
- `ৱ` aligned recall: **85.81%** over 458 occurrences
- Wall time: 266.0 seconds with four concurrent worker processes on a 2-core/4-thread Intel Xeon Platinum 8573C allocation

Raw machine-readable output is generated at `artifacts/tesseract/fast_test_metrics.json`. This baseline establishes that a generic Assamese language pack is useful but leaves substantial room for a purpose-trained recognizer.

