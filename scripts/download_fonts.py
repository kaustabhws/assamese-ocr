from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont

FONTS = {
    "NotoSansBengali.ttf": {
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansbengali/NotoSansBengali%5Bwdth%2Cwght%5D.ttf",
        "license": "OFL-1.1",
        "license_url": "https://openfontlicense.org/",
    },
    "NotoSerifBengali.ttf": {
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifbengali/NotoSerifBengali%5Bwdth%2Cwght%5D.ttf",
        "license": "OFL-1.1",
        "license_url": "https://openfontlicense.org/",
    },
    "AnekBangla.ttf": {
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/anekbangla/AnekBangla%5Bwdth%2Cwght%5D.ttf",
        "license": "OFL-1.1",
        "license_url": "https://openfontlicense.org/",
    },
    "BalooDa2.ttf": {
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/balooda2/BalooDa2%5Bwght%5D.ttf",
        "license": "OFL-1.1",
        "license_url": "https://openfontlicense.org/",
    },
}


def main() -> None:
    output = Path("assets/fonts")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "note": "Font names refer to the shared script; generated language content is Assamese only.",
        "fonts": {},
    }
    for filename, metadata in FONTS.items():
        destination = output / filename
        print(f"Downloading {filename}")
        with urllib.request.urlopen(metadata["url"], timeout=120) as response:
            data = response.read()
        destination.write_bytes(data)
        font = TTFont(destination)
        codepoints = {codepoint for table in font["cmap"].tables for codepoint in table.cmap}
        missing = [f"U+{value:04X}" for value in (0x09F0, 0x09F1) if value not in codepoints]
        if missing:
            destination.unlink()
            raise ValueError(f"{filename} lacks required Assamese glyphs: {', '.join(missing)}")
        manifest["fonts"][filename] = {
            **metadata,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (output / "font_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
