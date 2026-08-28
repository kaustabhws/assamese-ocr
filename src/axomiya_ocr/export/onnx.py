from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from axomiya_ocr.training.engine import model_from_checkpoint


def export_recognizer(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset: int = 17,
    quantize: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, vocab, checkpoint = model_from_checkpoint(checkpoint_path)
    example = torch.zeros(1, 1, 48, 256, dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "time"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    metadata = {
        "format_version": 1,
        "language": "as",
        "model": model.metadata(),
        "vocab": vocab.to_dict(),
        "input": {
            "name": "images",
            "dtype": "float32",
            "layout": "NCHW",
            "height": int(checkpoint["training_config"]["data"]["image_height"]),
            "max_width": int(checkpoint["training_config"]["data"]["max_image_width"]),
            "pad_to_max_width": True,
        },
        "output": {"name": "logits", "layout": "NTC", "blank_id": 0},
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_cer": checkpoint.get("best_cer"),
        "fp32_bytes": output_path.stat().st_size,
    }
    metadata_path = output_path.with_suffix(".json")
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = output_path.with_name(output_path.stem + ".int8.onnx")
        quantize_dynamic(
            str(output_path),
            str(quantized_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm", "LSTM"],
        )
        metadata["int8_path"] = str(quantized_path)
        metadata["int8_bytes"] = quantized_path.stat().st_size
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if quantize:
        quantized_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    _verify_onnx(output_path, model, example.numpy())
    return metadata


def _verify_onnx(output_path: Path, model: torch.nn.Module, sample: np.ndarray) -> None:
    import onnxruntime as ort

    with torch.inference_mode():
        torch_output = model(torch.from_numpy(sample)).detach().numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"images": sample})[0]
    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-4)
