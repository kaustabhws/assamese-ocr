from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

MODELS = {
    "fast": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/asm.traineddata",
    "best": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/asm.traineddata",
}


def main() -> None:
    root = Path("artifacts/tesseract")
    manifest: dict[str, object] = {"language": "asm", "models": {}}
    for profile, url in MODELS.items():
        output = root / profile / "asm.traineddata"
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Tesseract Assamese {profile}")
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
        output.write_bytes(data)
        manifest["models"][profile] = {
            "url": url,
            "path": str(output),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

