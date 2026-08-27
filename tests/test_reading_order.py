from axomiya_ocr.layout.reading_order import resolve_reading_order
from axomiya_ocr.layout.schema import BoundingBox, Region


def region(name: str, box: tuple[float, float, float, float]) -> Region:
    return Region(name, "text", BoundingBox(*box), 1.0)


def test_two_columns_follow_full_width_title() -> None:
    title = region("title", (50, 10, 950, 80))
    left_top = region("left-top", (50, 120, 450, 300))
    left_bottom = region("left-bottom", (50, 320, 450, 500))
    right_top = region("right-top", (550, 120, 950, 300))
    right_bottom = region("right-bottom", (550, 320, 950, 500))
    ordered = resolve_reading_order(
        [right_bottom, left_bottom, title, right_top, left_top], page_width=1000
    )
    assert [item.id for item in ordered] == [
        "title",
        "left-top",
        "left-bottom",
        "right-top",
        "right-bottom",
    ]

