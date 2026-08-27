from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

from axomiya_ocr.layout.schema import Document


def save_json(document: Document, output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=82)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def save_html(document: Document, page_images: list[Image.Image], output_path: str | Path) -> None:
    pages: list[str] = []
    for page, image in zip(document.pages, page_images, strict=True):
        line_elements: list[str] = []
        region_elements: list[str] = []
        for region in page.regions:
            box = region.bbox
            region_elements.append(
                f'<div class="region {html.escape(region.label)}" title="{html.escape(region.label)}" '
                f'style="left:{box.x0}px;top:{box.y0}px;width:{box.width}px;height:{box.height}px"></div>'
            )
            for line in region.lines:
                line_box = line.bbox
                font_size = max(8.0, line_box.height * 0.72)
                line_elements.append(
                    f'<span class="ocr-line" data-confidence="{line.confidence:.6f}" '
                    f'style="left:{line_box.x0}px;top:{line_box.y0}px;width:{line_box.width}px;'
                    f'height:{line_box.height}px;font-size:{font_size}px">{html.escape(line.text)}</span>'
                )
        pages.append(
            f'<section class="page" style="width:{page.width}px;height:{page.height}px">'
            f'<img src="{_data_url(image)}" alt="page {page.number}">' + "".join(region_elements) +
            "".join(line_elements) + "</section>"
        )
    markup = """<!doctype html>
<html lang="as"><head><meta charset="utf-8"><title>Assamese OCR</title>
<style>
body{margin:0;background:#333;font-family:"Noto Sans Bengali",sans-serif}.page{position:relative;margin:24px auto;background:white;box-shadow:0 2px 16px #0008}.page>img{position:absolute;width:100%;height:100%}.region{position:absolute;box-sizing:border-box;border:1px solid transparent}.page:hover .region{border-color:#2b7fff66}.ocr-line{position:absolute;color:transparent;white-space:pre;line-height:1;transform-origin:top left;user-select:text}.ocr-line::selection{background:#1687ff66;color:transparent}
</style></head><body>""" + "".join(pages) + "</body></html>"
    Path(output_path).write_text(markup, encoding="utf-8")


def save_searchable_pdf(
    document: Document,
    page_images: list[Image.Image],
    output_path: str | Path,
    font_path: str | Path,
) -> None:
    import fitz

    pdf = fitz.open()
    font_path = str(font_path)
    for page_data, image in zip(document.pages, page_images, strict=True):
        page = pdf.new_page(width=page_data.width, height=page_data.height)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        page.insert_image(page.rect, stream=buffer.getvalue())
        font_name = f"assamese{page_data.number}"
        page.insert_font(fontname=font_name, fontfile=font_path)
        for region in page_data.regions:
            for line in region.lines:
                if not line.text:
                    continue
                box = line.bbox
                rect = fitz.Rect(box.x0, box.y0, box.x1, box.y1)
                page.insert_textbox(
                    rect,
                    line.text,
                    fontname=font_name,
                    fontsize=max(4.0, box.height * 0.7),
                    render_mode=3,
                    overlay=True,
                )
    pdf.save(str(output_path), garbage=4, deflate=True)
