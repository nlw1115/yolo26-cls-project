from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from .detection_utils import predict_image
from .labelme_dataset import load_class_names, load_records
from .paths import DATASET_DIR, DERIVED_DIR
from .sam2_prompt_labelme import PromptPoint, Sam2PointLabeler, load_rgb_image


def model_label(model: YOLO, pred_cls: int, class_names: tuple[str, ...]) -> str:
    names = getattr(model.model, "names", None) or getattr(model, "names", None)
    if isinstance(names, dict) and pred_cls in names:
        return str(names[pred_cls])
    if 0 <= pred_cls < len(class_names):
        return class_names[pred_cls]
    return f"class_{pred_cls}"


def prelabel(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    weights = Path(args.weights).resolve()
    run_id = args.run_id or f"{weights.stem}_sam2_{args.conf:.2f}".replace(".", "p")
    output_dir = Path(args.output_root).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(dataset_dir / "classes.yaml")
    _, records, _ = load_records(dataset_dir=dataset_dir, subset=args.subset, class_mode="multi")
    model = YOLO(str(weights))
    sam2 = Sam2PointLabeler(model_name=args.sam2_model)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        predictions = predict_image(
            model=model,
            image_path=record.image_path,
            conf=args.conf,
            iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )
        image = load_rgb_image(record.image_path)
        shapes: list[dict[str, Any]] = []
        for pred in predictions:
            x = (float(pred["x1"]) + float(pred["x2"])) / 2.0
            y = (float(pred["y1"]) + float(pred["y2"])) / 2.0
            label = model_label(model, int(pred["pred_cls"]), class_names)
            prompt = PromptPoint(
                label=label,
                x=x,
                y=y,
                description=f"yolo_sam2: conf={float(pred['conf']):.4f}, center=({x:.1f},{y:.1f})",
                flags={"prelabel": True, "yolo": True, "sam2": True},
            )
            shape = sam2.shape_from_prompt(
                image=image,
                image_id=str(record.image_path),
                prompt=prompt,
                min_mask_area=args.min_mask_area,
            )
            if shape is not None:
                shapes.append(shape)
        with Image.open(record.image_path) as img:
            width, height = img.size
        payload = {
            "version": "6.1.3",
            "flags": {},
            "shapes": shapes,
            "imagePath": record.image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        json_path = output_dir / f"{record.stem}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(
            {
                "stem": record.stem,
                "image": str(record.image_path),
                "json": str(json_path),
                "detections": len(predictions),
                "sam2_shapes": len(shapes),
            }
        )
        print(f"[{index}/{len(records)}] {record.stem}: detections={len(predictions)}, sam2={len(shapes)}")
    (output_dir / "prelabel_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"预标注输出目录: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据集阶段：YOLO 粗检中心点调用 SAM2，输出待复核 LabelMe JSON。")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--subset", choices=["train", "val", "all"], default="all")
    parser.add_argument("--output-root", default=str(DERIVED_DIR / "prelabels"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sam2-model", default="sam2:tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--min-mask-area", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    prelabel(parse_args())

