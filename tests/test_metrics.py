from axomiya_ocr.training.metrics import OCRMetrics, character_recall, edit_distance


def test_edit_distance() -> None:
    assert edit_distance("অসম", "অসম") == 0
    assert edit_distance("অসম", "অম") == 1


def test_metrics() -> None:
    metrics = OCRMetrics()
    metrics.update("মই যাওঁ", "মই যাও")
    result = metrics.to_dict()
    assert result["samples"] == 1
    assert result["char_errors"] == 1
    assert result["sequence_accuracy"] == 0.0


def test_character_recall_uses_alignment_not_only_counts() -> None:
    result = character_recall(["ৰক"], ["কৰ"], "ৰ")
    assert result["reference_count"] == 1
    assert result["matched_count"] == 0
