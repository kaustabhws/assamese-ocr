from axomiya_ocr.layout.evaluation import bbox_iou, evaluate_documents


def test_bbox_iou() -> None:
    left = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    right = {"x0": 5, "y0": 0, "x1": 15, "y1": 10}
    assert bbox_iou(left, right) == 1 / 3


def test_document_metrics_include_layout_order_and_text() -> None:
    expected = {
        "pages": [
            {
                "number": 1,
                "regions": [
                    {
                        "label": "title",
                        "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
                        "order": 0,
                        "lines": [{"text": "অসম", "order": 0}],
                    },
                    {
                        "label": "text",
                        "bbox": {"x0": 0, "y0": 30, "x1": 100, "y1": 80},
                        "order": 1,
                        "lines": [{"text": "ৰ আৰু ৱ", "order": 0}],
                    },
                ],
            }
        ]
    }
    predicted = {
        "pages": [
            {
                "number": 1,
                "regions": [
                    {
                        "label": "title",
                        "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
                        "order": 0,
                        "lines": [{"text": "অসম", "order": 0}],
                    },
                    {
                        "label": "text",
                        "bbox": {"x0": 0, "y0": 30, "x1": 100, "y1": 80},
                        "order": 1,
                        "lines": [{"text": "ৰ আৰু ৱ", "order": 0}],
                    },
                ],
            }
        ]
    }
    result = evaluate_documents(expected, predicted)
    assert result["layout"]["f1"] == 1.0
    assert result["reading_order_pair_accuracy"] == 1.0
    assert result["ocr"]["cer"] == 0.0

