import json

from PIL import Image

from axomiya_ocr.export.document import save_html, save_json
from axomiya_ocr.inference.document import DocumentOCR
from axomiya_ocr.layout.schema import BoundingBox, Region


class FakeLayout:
    def predict(self, image):
        return [Region("body", "text", BoundingBox(5, 5, image.width - 5, 50), 0.9)]


class FakeDetector:
    def predict(self, image):
        return [[[10, 10], [90, 10], [90, 35], [10, 35]]]


class FakeRecognizer:
    def predict(self, image):
        return "অসমীয়া", 0.99


def test_document_pipeline_and_exports(tmp_path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(source)
    pipeline = DocumentOCR(FakeLayout(), FakeRecognizer(), FakeDetector())
    document, images = pipeline.process(source)
    assert document.language == "as"
    assert document.pages[0].regions[0].lines[0].text == "অসমীয়া"

    json_path = tmp_path / "document.json"
    html_path = tmp_path / "document.html"
    save_json(document, json_path)
    save_html(document, images, html_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["language"] == "as"
    assert "অসমীয়া" in html_path.read_text(encoding="utf-8")

