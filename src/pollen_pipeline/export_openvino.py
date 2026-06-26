from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import onnx
from openvino import save_model
from openvino.tools import ovc
from ultralytics import YOLO

from .labelme_dataset import load_class_names
from .paths import DATASET_DIR, RUNS_DIR


def set_metadata(ov_model, metadata: dict[str, object]) -> None:
    for key, value in metadata.items():
        ov_model.set_rt_info(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value, ["pollen_pipeline", key])


def export(args: argparse.Namespace) -> None:
    weights = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(Path(args.dataset_dir).resolve() / "classes.yaml")
    if args.class_mode == "single":
        class_names = (args.single_class_name,)
    model = YOLO(str(weights))
    with tempfile.TemporaryDirectory(prefix="pollen_export_") as tmp_name:
        tmp_dir = Path(tmp_name)
        onnx_path = Path(
            model.export(
                format="onnx",
                half=False,
                imgsz=args.imgsz,
                batch=args.batch,
                opset=args.opset,
                simplify=args.simplify,
                dynamic=args.dynamic,
                device=args.device,
            )
        )
        local_onnx = tmp_dir / "model_fp16.onnx"
        shutil.copy2(onnx_path, local_onnx)
        onnx.checker.check_model(str(local_onnx))
        ov_model = ovc.convert_model(str(local_onnx))
        fp16 = args.fp16 or args.precision == "fp16"
        metadata = {
            "task": "pollen_detector",
            "weights": str(weights),
            "class_mode": args.class_mode,
            "class_names": list(class_names),
            "input_width": args.imgsz,
            "input_height": args.imgsz,
            "batch": args.batch,
            "dynamic": args.dynamic,
            "input_name": "images",
            "output_name": "output0",
            "precision": "fp16" if fp16 else "fp32",
        }
        set_metadata(ov_model, metadata)
        xml_path = output_dir / "model.xml"
        save_model(ov_model, str(xml_path), compress_to_fp16=fp16)
        if args.keep_onnx:
            shutil.copy2(local_onnx, output_dir / "model_fp16.onnx")
        elif onnx_path.exists() and onnx_path.resolve() != local_onnx.resolve():
            onnx_path.unlink()
    (output_dir / "export_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OpenVINO 导出完成: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出检测模型为 OpenVINO，并把类别/输入尺寸写入模型 metadata。")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "export_openvino"))
    parser.add_argument("--class-mode", choices=["single", "multi"], default="single")
    parser.add_argument("--single-class-name", default="pollen")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", help="导出动态 ONNX/OpenVINO。默认保持静态 batch=1 生产路线。")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--fp16", action="store_true", help="兼容旧入口；等价于 --precision fp16。")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--keep-onnx", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
