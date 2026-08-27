# Assamese CRNN-CTC model card

## Model description

- Language: Assamese (`as`)
- Task: printed word/line recognition
- Architecture: compact convolutional encoder + 2-layer BiLSTM + CTC
- Input: grayscale, height 48, dynamic width
- Output: NFC Unicode code points from the checked-in Assamese repertoire
- Intended use: OCR crops produced by the bundled text detector
- Out of scope: handwriting and Bengali-language OCR

## Training data

Fill in exact source revisions, sample counts, exclusion counts, font hashes, synthetic proportion, and the applicable licenses from the generated audit files.

## Results

Do not publish this card until all fields below are measured.

| Metric | FP32 | INT8 |
|---|---:|---:|
| Mozhi test CER | TBD | TBD |
| Mozhi test WER | TBD | TBD |
| Exact sequence accuracy | TBD | TBD |
| `ৰ` recall | TBD | TBD |
| `ৱ` recall | TBD | TBD |
| Real-page end-to-end CER | TBD | TBD |
| Median CPU latency | TBD | TBD |
| p95 CPU latency | TBD | TBD |
| Model size | TBD | TBD |

## Limitations

Document all observed failures by font, scan quality, document period, mixed-language content, and semantic region. Word-level Mozhi metrics must not be described as page-level performance.

