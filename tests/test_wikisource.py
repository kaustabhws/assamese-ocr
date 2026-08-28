from axomiya_ocr.data.wikisource import (
    align_transcription,
    document_split,
    page_text_from_html,
    select_documents,
)


def test_page_text_parser_keeps_only_proofread_body() -> None:
    markup = (
        '<div class="quality4">UI header</div>'
        '<div class="pagetext"><p>অসমীয়া ভাষা</p><p>আৰু সাহিত্য।</p></div>'
        '<div>UI footer</div>'
    )
    assert page_text_from_html(markup) == "অসমীয়া ভাষা আৰু সাহিত্য।"


def test_alignment_assigns_corrected_reference_to_each_line() -> None:
    aligned = align_transcription(
        ["অসমীয়া ভাযা", "আৰু সাহিত্য"],
        "অসমীয়া ভাষা আৰু সাহিত্য।",
    )
    assert [line.transcription for line in aligned] == ["অসমীয়া ভাষা", "আৰু সাহিত্য।"]
    assert aligned[0].alignment_score > 0.8


def test_document_selection_is_deterministic_and_disjoint() -> None:
    titles = [f"সূচী:book-{index}.pdf" for index in range(100)]
    first = select_documents(titles, {"train": 5, "validation": 2, "test": 2}, seed=9)
    second = select_documents(reversed(titles), {"train": 5, "validation": 2, "test": 2}, seed=9)
    assert first == second
    flattened = [title for values in first.values() for title in values]
    assert len(flattened) == len(set(flattened))
    for split, values in first.items():
        assert all(document_split(title, 9) == split for title in values)

