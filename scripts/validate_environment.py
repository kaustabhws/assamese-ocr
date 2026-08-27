from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


def main() -> None:
    optional = ("torch", "datasets", "onnxruntime", "cv2", "fitz", "rapidocr", "jupyterlab")
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "workspace": str(Path.cwd()),
        "packages": {name: importlib.util.find_spec(name) is not None for name in optional},
    }
    if report["packages"]["torch"]:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

