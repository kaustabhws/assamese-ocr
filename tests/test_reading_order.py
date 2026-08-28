from axomiya_ocr.layout.reading_order import (
    assign_lines_to_regions,
    deduplicate_regions,
    resolve_reading_order,
)
from axomiya_ocr.layout.schema import BoundingBox, Region, TextLine


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


def line(name: str, box: tuple[float, float, float, float]) -> TextLine:
    bbox = BoundingBox(*box)
    return TextLine(name, bbox, [], name, 1.0)


def test_near_identical_regions_keep_highest_confidence() -> None:
    low = Region("low", "text", BoundingBox(10, 10, 100, 50), 0.4)
    high = Region("high", "section_header", BoundingBox(10, 10, 100, 50), 0.9)
    assert [item.id for item in deduplicate_regions([low, high])] == ["high"]


def test_table_container_receives_lines_instead_of_line_sized_regions() -> None:
    table = Region("table", "key_value_region", BoundingBox(10, 100, 900, 900), 0.9)
    tiny = Region("tiny", "text", BoundingBox(20, 120, 200, 150), 0.8)
    assigned = assign_lines_to_regions([table, tiny], [line("entry", (20, 120, 200, 150))])
    assert [item.id for item in assigned] == ["table"]
    assert assigned[0].lines[0].id == "entry"


def test_reading_order_uses_line_content_not_oversized_region_box() -> None:
    heading = region("heading", (100, 200, 400, 240))
    heading.lines = [line("heading-line", (100, 200, 400, 240))]
    table = region("table", (50, 100, 950, 900))
    table.label = "key_value_region"
    table.lines = [line("table-line", (100, 300, 300, 330))]
    ordered = resolve_reading_order([table, heading], page_width=1000)
    assert [item.id for item in ordered] == ["heading", "table"]


def test_small_header_band_is_top_to_bottom_not_false_columns() -> None:
    left_lower = region("left-lower", (100, 120, 300, 150))
    right_upper = region("right-upper", (600, 50, 900, 80))
    ordered = resolve_reading_order([left_lower, right_upper], page_width=1000)
    assert [item.id for item in ordered] == ["right-upper", "left-lower"]


def test_page_top_band_precedes_lower_left_column() -> None:
    header = region("header", (550, 220, 900, 250))
    lower_left = region("lower-left", (100, 350, 400, 390))
    body_left = region("body-left", (100, 500, 400, 800))
    ordered = resolve_reading_order(
        [lower_left, body_left, header], page_width=1000, page_height=1900
    )
    assert ordered[0].id == "header"


def test_latin_qr_noise_inside_picture_is_not_exported_as_text() -> None:
    picture = Region("qr", "picture", BoundingBox(0, 0, 200, 200), 0.9)
    assigned = assign_lines_to_regions([picture], [line("QR-12X", (20, 20, 180, 50))])
    assert assigned == [picture]
    assert picture.lines == []


def test_mixed_qr_noise_with_small_assamese_fragment_is_not_exported() -> None:
    picture = Region("qr", "picture", BoundingBox(0, 0, 200, 200), 0.9)
    assigned = assign_lines_to_regions([picture], [line("C 3! 2% প্ৰ", (20, 20, 180, 50))])
    assert assigned == [picture]
    assert picture.lines == []

