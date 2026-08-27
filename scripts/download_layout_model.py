from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description="Download the balanced INT8 layout model")
    parser.add_argument("--output", default="artifacts/layout")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id="stefanj0/docling-layout-heron-int8-onnx",
            filename="docling-layout-heron-int8.onnx",
            revision="dc6c61cf80f3771f3224c59e85b921310802682f",
        )
    )
    destination = output / "docling-layout-heron-int8.onnx"
    destination.write_bytes(cached.read_bytes())
    print(destination)


if __name__ == "__main__":
    main()
