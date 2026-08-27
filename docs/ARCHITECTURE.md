# Architecture

## Why this is a pipeline

A recognition network turns a text crop into characters; it does not know that a crop is a title, table cell, caption, or second-column paragraph. Preserving a page therefore requires separate spatial and linguistic stages with one stable output contract.

```text
image/PDF page
    -> render, orient, deskew
    -> semantic layout detector
    -> text-line detector inside readable regions
    -> crop rectification
    -> Assamese CRNN-CTC recognizer
    -> reading-order resolver
    -> canonical document JSON
    -> HTML / searchable PDF / downstream adapters
```

## Recognition model

The recognizer follows the strongest family reported in *Towards Deployable OCR Models for Indic Languages*: a convolutional visual encoder, bidirectional recurrent sequence encoder, and CTC transcription. This implementation uses depth-efficient convolutions, keeps horizontal resolution, accepts dynamic widths, and has no autoregressive decoder. Those choices make ONNX export and CPU inference predictable.

Input is a grayscale crop normalized to height 48. Width is proportional to the original aspect ratio and padded within the batch. The network emits one distribution approximately every four input pixels. Training rejects samples that cannot satisfy the CTC alignment constraint at the configured maximum width rather than silently truncating their labels.

## Assamese boundary

- Accepted OCR source: `darknight054/indic-mozhi-ocr`, exact config `assamese`.
- Accepted synthetic text source: Assamese Wikipedia, exact config `20231101.as`, or user-provided Assamese text.
- Bengali config `bengali` and Bengali Wikipedia config `20231101.bn` are never selected.
- Unicode is NFC-normalized without transliteration or Bengali-to-Assamese rewriting.
- The audit reports Assamese-specific `ৰ` (U+09F0) and `ৱ` (U+09F1) coverage.

The script is shared, so a generic per-line Unicode heuristic cannot reliably distinguish every Assamese sentence from Bengali. Source provenance is therefore enforced in addition to character-level checks.

## Layout profile

The balanced path uses the Apache-2.0 Docling Heron INT8 ONNX detector. It accepts a uint8 RGB tensor at 640x640 and returns 17 semantic region classes. The mobile training notebook prepares an SSDLite-MobileNetV3 student for DocLayNet; it must be benchmarked before replacing Heron.

## Output contract

Every page has pixel dimensions and ordered regions. Every region has a semantic label, bounding box, confidence, and ordered lines. Every text line has a polygon, UTF-8 text, recognition confidence, and optional character spans. Coordinates remain in original page pixels, which permits faithful visual overlays and later PDF reconstruction.

