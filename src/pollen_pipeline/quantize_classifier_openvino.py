from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import nncf
import numpy as np
from openvino import Core, save_model

from .crop import crop_letterbox_from_bbox
from .labelme_dataset import load_records
from .paths import DATASET_DIR, RUNS_DIR


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return np.transpose(rgb, (2, 0, 1))[None, :, :, :].astype(np.float32)


def build_calibration_tensors(
    dataset_dir: Path,
    image_size: int,
    crop_scale: float,
    samples_per_class: int,
) -> list[np.ndarray]:
    _, records, _ = load_records(dataset_dir=dataset_dir, subset="train", class_mode="multi")
    records = sorted(records, key=lambda record: record.stem)
    counts: dict[int, int] = {}
    tensors: list[np.ndarray] = []
    for record in records:
        image = read_image(record.image_path)
        for box in record.boxes:
            cls_id = int(box.original_cls)
            if samples_per_class > 0 and counts.get(cls_id, 0) >= samples_per_class:
                continue
            crop = crop_letterbox_from_bbox(image, box.bbox_xyxy, image_size=image_size, crop_scale=crop_scale)
            tensors.append(preprocess_crop(crop))
            counts[cls_id] = counts.get(cls_id, 0) + 1
    if not tensors:
        raise ValueError("没有生成分类器 INT8 校准样本，请检查 dataset/split.json 和 LabelMe 标注。")
    return tensors


def quantize(args: argparse.Namespace) -> None:
    model_xml = Path(args.model_xml).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_tensors = build_calibration_tensors(
        dataset_dir=dataset_dir,
        image_size=args.image_size,
        crop_scale=args.crop_scale,
        samples_per_class=args.samples_per_class,
    )
    core = Core()
    model = core.read_model(str(model_xml))
    dataset = nncf.Dataset(calibration_tensors)
    quantized = nncf.quantize(
        model,
        dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=min(args.subset_size, len(calibration_tensors)),
    )
    xml_path = output_dir / "model_int8.xml"
    save_model(quantized, str(xml_path), compress_to_fp16=False)

    metadata = {
        "task": "pollen_classifier_int8_ptq",
        "source_model_xml": str(model_xml),
        "output_model_xml": str(xml_path),
        "dataset_dir": str(dataset_dir),
        "calibration_samples": len(calibration_tensors),
        "subset_size": min(args.subset_size, len(calibration_tensors)),
        "samples_per_class": args.samples_per_class,
        "image_size": args.image_size,
        "crop_scale": args.crop_scale,
        "precision": "int8",
    }
    (output_dir / "quantize_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"分类器 INT8 PTQ 完成: {xml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对已导出的分类器 OpenVINO 模型执行 INT8 PTQ。")
    parser.add_argument("--model-xml", required=True, help="分类器 FP32/FP16 OpenVINO model.xml。")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "export_openvino_classifier_int8"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=1.7)
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--subset-size", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    quantize(parse_args())
