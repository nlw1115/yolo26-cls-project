from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from openvino import save_model
from openvino.tools import ovc

from .classifier_train import SUPPORTED_MODELS, build_model, resolve_device
from .export_openvino import set_metadata
from .paths import RUNS_DIR


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"分类权重格式不正确，缺少 state_dict: {path}")
    return checkpoint


def export_onnx(model: torch.nn.Module, dummy: torch.Tensor, output_path: Path, opset: int) -> None:
    kwargs = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamo": False,
    }
    try:
        torch.onnx.export(model, dummy, str(output_path), **kwargs)
    except TypeError:
        kwargs.pop("dynamo", None)
        torch.onnx.export(model, dummy, str(output_path), **kwargs)


def export(args: argparse.Namespace) -> None:
    weights = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(weights, device)
    model_name = args.model or str(checkpoint.get("model_name") or "resnet34")
    num_classes = int(checkpoint.get("num_classes") or len(checkpoint.get("class_names") or []))
    if num_classes <= 0:
        raise ValueError("分类权重中无法解析 num_classes/class_names")

    input_size = int(args.image_size or checkpoint.get("input_size") or 224)
    class_names = [str(name) for name in checkpoint.get("class_names", [f"class_{i}" for i in range(num_classes)])]

    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    metadata = {
        "task": "pollen_classifier",
        "weights": str(weights),
        "model_name": model_name,
        "class_names": class_names,
        "class_to_idx": checkpoint.get("class_to_idx", {}),
        "input_width": input_size,
        "input_height": input_size,
        "input_name": "images",
        "output_name": "logits",
        "precision": "fp16" if args.fp16 else "fp32",
        "preprocessing": checkpoint.get("preprocessing", {}),
    }

    with tempfile.TemporaryDirectory(prefix="pollen_cls_export_") as tmp_name:
        tmp_dir = Path(tmp_name)
        local_onnx = tmp_dir / "classifier_fp16.onnx"
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        export_onnx(model, dummy, local_onnx, args.opset)
        onnx.checker.check_model(str(local_onnx))
        ov_model = ovc.convert_model(str(local_onnx))
        set_metadata(ov_model, metadata)
        save_model(ov_model, str(output_dir / "model.xml"), compress_to_fp16=args.fp16)
        if args.keep_onnx:
            shutil.copy2(local_onnx, output_dir / "classifier_fp16.onnx")

    (output_dir / "export_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"分类 OpenVINO 导出完成: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 ResNet/EfficientNet 分类模型为 OpenVINO FP16，并写入类别和输入尺寸 metadata。")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "export_openvino_classifier"))
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--keep-onnx", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
