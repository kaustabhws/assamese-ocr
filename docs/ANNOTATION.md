# Assamese page benchmark annotation

Recognition training can begin with Mozhi and synthetic lines, but end-to-end page quality needs a manually checked Assamese benchmark. Keep this benchmark private until every source's redistribution rights are known.

## Sampling

- Select at least 10 independent books/newspapers/forms and at least 100 pages total.
- Store a stable document ID, page number, source URL, rights statement, and SHA-256.
- Split by document, never by page or text crop.
- Ensure old and modern typefaces, one/two-column pages, tables, pictures, and degraded scans are represented.
- Track the count of `ৰ` and `ৱ`; neither character may be absent from the test set.

## Labels

Draw semantic regions first, then line polygons inside readable regions. Transcribe exactly what is printed using NFC Unicode; preserve punctuation and do not modernize spelling. Mark illegible spans instead of guessing.

The Label Studio configuration in `annotation/label_studio.xml` defines the semantic classes. Export annotations as JSON, convert coordinates to original-page pixels, and validate them against `schemas/document.schema.json`.

After inference, run `python scripts/evaluate_document.py --ground-truth path/to/gold.json --prediction artifacts/demo/document.json`. The report includes layout precision/recall/F1 at IoU 0.5, reading-order pair accuracy, page CER/WER, and Assamese-specific character recall.

## Quality control

- A second Assamese reader reviews every transcription in validation and test.
- Resolve disagreements while retaining the original versions in annotation history.
- Run Unicode normalization and unknown-character reports.
- Visually overlay boxes and reading-order numbers before freezing the test set.
