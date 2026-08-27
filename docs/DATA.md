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

## Preparation rules

1. Preserve the official Mozhi train/validation/test assignment; remove exact pixel duplicates from validation/test rather than moving examples between splits.
2. Normalize labels with Unicode NFC and collapse spacing; do not rewrite characters.
3. Reject empty labels, control characters, excessive non-Assamese letters, corrupt images, and impossible CTC alignments.
4. Build the vocabulary from the training split plus explicit Assamese-required characters. Report—but never silently absorb—unknown validation/test symbols.
5. Hash source image bytes and normalized text to find exact cross-split leakage.
6. For user page data, split by book/document before creating crops. Pages from one book must not leak across train and test.
7. Keep the official test split untouched. Tune thresholds only on validation.

## Required real-page benchmark

Before calling the project accurate, create a small gold set covering at least:

- modern clean print
- older typefaces
- skewed and photographed pages
- two-column newspaper pages
- tables/forms
- pages containing `ৰ` and `ৱ`

Recommended minimum: 100 pages from at least 10 source documents, with document-level splits and independently checked transcriptions. The repository includes the canonical annotation schema used by the reconstruction code.
