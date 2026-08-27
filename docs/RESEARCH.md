# Research and dependency decisions

Checked on 2026-08-27. URLs are recorded so model cards can cite the exact upstream artifacts.

## Recognition

- [Mozhi paper](https://arxiv.org/abs/2205.06740): its experiments found CRNN + CTC strongest among the tested compact recognition encoders for nearly every Indic language. It also distinguishes word, line, and end-to-end page evaluation. This is the basis for the custom recognizer and metric separation here.
- [Mozhi Assamese mirror](https://huggingface.co/datasets/darknight054/indic-mozhi-ocr): 79,697 train, 9,945 validation, and 10,146 test word images before validation filtering. The preparation code pins the downloaded revision and hard-codes the `assamese` config.
- [MWirelabs Assamese OCR](https://huggingface.co/MWirelabs/assamese-ocr): a useful research reference, but its model card describes a 378M-parameter Florence-2-based system with an unverified 5.33% CER. It is not used because it conflicts with the lightweight CPU target and is less accurate than the compact CRNN result reported by the Mozhi paper.
- [PP-OCRv5 multilingual documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md): lists 106 supported languages but, at the checked revision, does not list Assamese or Bengali. Its script-independent mobile detector remains useful; its recognizer is not treated as an Assamese model.
- [Tesseract Assamese models](https://github.com/tesseract-ocr/tessdata_fast/blob/main/asm.traineddata): both `fast` and `best` become measured baselines, never assumed targets.

## Layout

- [Docling Heron](https://huggingface.co/docling-project/docling-layout-heron): Apache-2.0, 17 semantic classes.
- [Heron ONNX](https://huggingface.co/docling-project/docling-layout-heron-onnx): official 171 MB FP32 export with a simple uint8 input contract.
- [Heron INT8 ONNX](https://huggingface.co/stefanj0/docling-layout-heron-int8-onnx): 69 MB community quantization. Its card reports near-lossless agreement with FP32 on 88 pages, not independent ground-truth mAP; this repository repeats evaluation on Assamese pages before release.
- [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2): 80,863 human-annotated pages and 11 layout classes under CDLA-Permissive-1.0. It is suitable for a future small layout student, but the enriched mirror is roughly 40 GB, so it is not silently downloaded by the recognizer notebooks.

## Why Bengali data is excluded

Assamese and Bengali share the Unicode block that Unicode names `Bengali`. Script block names, fonts named `*Bengali`, and Unicode character names are therefore not evidence that Bengali-language text entered training. Provenance is the enforceable boundary: only Mozhi `assamese` and Wikipedia `20231101.as` are accepted. Any cross-language transfer experiment must use a separate config and report separately.

