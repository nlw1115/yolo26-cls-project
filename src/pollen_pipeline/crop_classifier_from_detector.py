from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .common import write_csv, write_json
from .crop import crop_letterbox_from_bbox, imread_unicode, imwrite_unicode
from .detection_utils import match_predictions, predict_image
from .labelme_dataset import load_class_names, load_records
from .paths import DATASET_DIR, DERIVED_DIR


def safe_run_id(weights: Path, conf: float, imgsz: int, crop_scale: float) -> str:
    return f"{weights.stem}_conf{conf:.2f}_img{imgsz}_crop{crop_scale:.2f}".replace(".", "p")


def write_classes_json(output_dir: Path, class_names: tuple[str, ...]) -> None:
    write_json(
        output_dir / "classes.json",
        {
            "classes": [
                {"id": idx, "folder": f"class_{idx}", "name": name}
                for idx, name in enumerate(class_names)
            ]
        },
    )


def crop_from_detector(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    weights = Path(args.weights).resolve()
    run_id = args.run_id or safe_run_id(weights, args.conf, args.imgsz, args.crop_scale)
    pred_dir = Path(args.predictions_dir).resolve() / run_id
    crop_dir = Path(args.output_root).resolve() / run_id
    class_names = load_class_names(dataset_dir / "classes.yaml")
    model = YOLO(str(weights))
    prediction_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    totals = {"matched_crops": 0, "unmatched_predictions": 0, "missed_gt": 0}
    for subset in ("train", "val"):
        _, records, _ = load_records(dataset_dir=dataset_dir, subset=subset, class_mode="multi")
        for record in records:
            image = imread_unicode(record.image_path)
            predictions = predict_image(
                model=model,
                image_path=record.image_path,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                max_det=args.max_det,
            )
            matched_rows, matched_gt = match_predictions(
                predictions=predictions,
                gt_boxes=record.boxes,
                match_iou=args.match_iou,
                class_aware=False,
            )
            totals["unmatched_predictions"] += sum(1 for row in matched_rows if not row["matched"])
            totals["missed_gt"] += len(record.boxes) - len(matched_gt)
            for row in matched_rows:
                prediction_rows.append({"subset": subset, "stem": record.stem, **row})
                if not row["matched"]:
                    continue
                cls_id = int(row["gt_cls"])
                class_folder = f"class_{cls_id}"
                crop = crop_letterbox_from_bbox(
                    image,
                    [row["x1"], row["y1"], row["x2"], row["y2"]],
                    crop_scale=args.crop_scale,
                    image_size=args.image_size,
                )
                file_name = f"{record.stem}_pred{int(row['pred_index']):03d}_gt{int(row['gt_index']):03d}.jpg"
                rel_path = Path(subset) / class_folder / file_name
                out_path = crop_dir / rel_path
                imwrite_unicode(out_path, crop, quality=args.jpeg_quality)
                totals["matched_crops"] += 1
                crop_rows.append(
                    {
                        "image": rel_path.as_posix(),
                        "subset": subset,
                        "stem": record.stem,
                        "class_id": cls_id,
                        "class_name": class_names[cls_id],
                        "source_image": str(record.image_path),
                        "detector_weights": str(weights),
                        "pred_conf": row["conf"],
                        "match_iou": row["match_iou"],
                        "x1": row["x1"],
                        "y1": row["y1"],
                        "x2": row["x2"],
                        "y2": row["y2"],
                    }
                )
    write_csv(pred_dir / "manifest.csv", prediction_rows)
    write_csv(crop_dir / "manifest.csv", crop_rows)
    write_classes_json(crop_dir, class_names)
    write_json(
        crop_dir / "crop_meta.json",
        {
            "run_id": run_id,
            "dataset_dir": str(dataset_dir),
            "detector_weights": str(weights),
            "conf": args.conf,
            "iou": args.iou,
            "match_iou": args.match_iou,
            "imgsz": args.imgsz,
            "crop_scale": args.crop_scale,
            "image_size": args.image_size,
            "classes": list(class_names),
            "totals": totals,
            "note": "分类数据只保存检测框与 GT 匹配成功的 crop；未匹配预测框和漏检只写入预测清单。",
        },
    )
    print(f"检测推理清单: {pred_dir / 'manifest.csv'}")
    print(f"分类离线 crop: {crop_dir}")
    print(totals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用指定检测权重全量推理，再离线裁剪分类训练图片。")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--predictions-dir", default=str(DERIVED_DIR / "detector_predictions"))
    parser.add_argument("--output-root", default=str(DERIVED_DIR / "classification_crops"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.6, help="Ultralytics predict iou 参数；YOLO26 end-to-end 导出不会在部署 demo 中额外做 NMS。")
    parser.add_argument("--nms-iou", dest="iou", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    crop_from_detector(parse_args())
