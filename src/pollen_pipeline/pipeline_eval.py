from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .classifier_train import build_model, build_transforms, resolve_device
from .common import write_csv, write_json
from .crop import crop_letterbox_from_bbox, imread_unicode
from .detector_eval import class_name, compute_ap_by_class, round_metric
from .detection_utils import match_predictions, predict_image
from .labelme_dataset import load_class_names, load_records
from .paths import DATASET_DIR, RUNS_DIR


def load_classifier(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"], int(checkpoint["num_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    _, val_tf = build_transforms(int(checkpoint.get("input_size", 224)), augment=False)
    return model, checkpoint, val_tf


def classify_crop(model: torch.nn.Module, transform: Any, crop_bgr: np.ndarray, device: torch.device) -> tuple[int, float]:
    rgb = crop_bgr[:, :, ::-1].copy()
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


def write_confusion(path: Path, matrix: np.ndarray, class_names: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt\\pred", *class_names])
        for idx, row in enumerate(matrix.tolist()):
            writer.writerow([class_names[idx] if idx < len(class_names) else f"class_{idx}", *row])


def plot_confusion(path: Path, matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, str(int(matrix[y, x])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(Path(args.dataset_dir).resolve() / "classes.yaml")
    _, records, _ = load_records(dataset_dir=Path(args.dataset_dir), subset=args.subset, class_mode="multi")
    detector = YOLO(str(Path(args.detector_weights).resolve()))
    classifier_device = resolve_device(args.device)
    classifier, checkpoint, transform = load_classifier(Path(args.classifier_weights).resolve(), classifier_device)
    num_classes = int(checkpoint["num_classes"])
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    predictions_for_ap: dict[str, list[dict[str, Any]]] = {}
    per_class_totals: dict[int, dict[str, int]] = {
        cls_id: {"gt": 0, "pred": 0, "tp": 0} for cls_id in range(len(class_names))
    }
    detector_totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    cascade_totals = {"images": 0, "gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "class_ok": 0}
    for record in records:
        image = imread_unicode(record.image_path)
        raw_predictions = predict_image(
            detector,
            record.image_path,
            min(args.conf, args.ap_conf),
            args.nms_iou,
            args.imgsz,
            args.device,
            args.max_det,
        )
        classified_predictions: list[dict[str, Any]] = []
        for row in raw_predictions:
            crop = crop_letterbox_from_bbox(
                image,
                [row["x1"], row["y1"], row["x2"], row["y2"]],
                crop_scale=args.crop_scale,
                image_size=args.image_size,
            )
            pred_cls, pred_score = classify_crop(classifier, transform, crop, classifier_device)
            detector_conf = float(row["conf"])
            classified_predictions.append(
                {
                    **row,
                    "detector_cls": int(row["pred_cls"]),
                    "detector_conf": detector_conf,
                    "pred_cls": pred_cls,
                    "classified_as": pred_cls,
                    "class_score": pred_score,
                    "conf": detector_conf * pred_score,
                }
            )

        predictions_for_ap[record.stem] = [
            {"stem": record.stem, **row}
            for row in classified_predictions
            if float(row["detector_conf"]) >= args.ap_conf
        ]
        predictions = [row for row in classified_predictions if float(row["detector_conf"]) >= args.conf]
        detector_matched_rows, detector_matched_gt = match_predictions(
            predictions=predictions,
            gt_boxes=record.boxes,
            match_iou=args.match_iou,
            class_aware=False,
        )
        cascade_matched_rows, cascade_matched_gt = match_predictions(
            predictions=predictions,
            gt_boxes=record.boxes,
            match_iou=args.match_iou,
            class_aware=True,
        )

        by_pred_index = {int(row["pred_index"]): row for row in cascade_matched_rows}
        for row in detector_matched_rows:
            cascade_row = by_pred_index.get(int(row["pred_index"]), row)
            correct = False
            if row["matched"]:
                gt_cls = int(row["gt_cls"])
                pred_cls = int(row["pred_cls"])
                confusion[gt_cls, pred_cls] += 1
                correct = gt_cls == pred_cls
                cascade_totals["class_ok"] += int(correct)
            rows.append(
                {
                    "stem": record.stem,
                    **row,
                    "cascade_matched": bool(cascade_row.get("matched", False)),
                    "cascade_gt_index": cascade_row.get("gt_index", ""),
                    "cascade_match_iou": cascade_row.get("match_iou", 0.0),
                    "class_score": row["class_score"],
                    "class_correct": int(correct),
                }
            )

        for box in record.boxes:
            per_class_totals.setdefault(int(box.cls), {"gt": 0, "pred": 0, "tp": 0})["gt"] += 1
        for pred in predictions:
            per_class_totals.setdefault(int(pred["pred_cls"]), {"gt": 0, "pred": 0, "tp": 0})["pred"] += 1
        for row in cascade_matched_rows:
            if row["matched"] and row["gt_cls"] != "":
                per_class_totals.setdefault(int(row["gt_cls"]), {"gt": 0, "pred": 0, "tp": 0})["tp"] += 1

        detector_totals["gt"] += len(record.boxes)
        detector_totals["pred"] += len(predictions)
        detector_totals["tp"] += len(detector_matched_gt)
        detector_totals["fp"] += len(predictions) - len(detector_matched_gt)
        detector_totals["fn"] += len(record.boxes) - len(detector_matched_gt)
        cascade_totals["images"] += 1
        cascade_totals["gt"] += len(record.boxes)
        cascade_totals["pred"] += len(predictions)
        cascade_totals["tp"] += len(cascade_matched_gt)
        cascade_totals["fp"] += len(predictions) - len(cascade_matched_gt)
        cascade_totals["fn"] += len(record.boxes) - len(cascade_matched_gt)

    write_csv(output_dir / "pipeline_predictions.csv", rows)
    write_confusion(output_dir / "pipeline_confusion_matrix.csv", confusion, class_names)
    plot_confusion(output_dir / "pipeline_confusion_matrix.png", confusion)
    detector_precision = detector_totals["tp"] / max(detector_totals["tp"] + detector_totals["fp"], 1)
    detector_recall = detector_totals["tp"] / max(detector_totals["tp"] + detector_totals["fn"], 1)
    cascade_precision = cascade_totals["tp"] / max(cascade_totals["tp"] + cascade_totals["fp"], 1)
    cascade_recall = cascade_totals["tp"] / max(cascade_totals["tp"] + cascade_totals["fn"], 1)
    matched_class_acc = cascade_totals["class_ok"] / max(detector_totals["tp"], 1)

    ap_by_class = compute_ap_by_class(records, predictions_for_ap, list(per_class_totals))
    per_class_rows: list[dict[str, Any]] = []
    for cls_id in sorted(per_class_totals):
        class_total = per_class_totals[cls_id]
        class_tp = class_total["tp"]
        class_fp = class_total["pred"] - class_tp
        class_fn = class_total["gt"] - class_tp
        class_precision = class_tp / max(class_tp + class_fp, 1)
        class_recall = class_tp / max(class_tp + class_fn, 1)
        class_f1 = 2.0 * class_precision * class_recall / max(class_precision + class_recall, 1e-12)
        ap_metrics = ap_by_class.get(cls_id, {"ap50": None, "map50_95": None})
        per_class_rows.append(
            {
                "class_id": cls_id,
                "class_name": class_name(class_names, cls_id),
                "gt": class_total["gt"],
                "pred": class_total["pred"],
                "tp": class_tp,
                "fp": class_fp,
                "fn": class_fn,
                "precision": round_metric(class_precision),
                "recall": round_metric(class_recall),
                "f1": round_metric(class_f1),
                "ap50": round_metric(ap_metrics["ap50"]),
                "map50_95": round_metric(ap_metrics["map50_95"]),
            }
        )
    valid_class_rows = [row for row in per_class_rows if int(row["gt"]) > 0]
    valid_ap50 = [float(row["ap50"]) for row in valid_class_rows if row["ap50"] != ""]
    valid_map = [float(row["map50_95"]) for row in valid_class_rows if row["map50_95"] != ""]
    macro_ap50 = float(np.mean(valid_ap50)) if valid_ap50 else 0.0
    macro_map50_95 = float(np.mean(valid_map)) if valid_map else 0.0

    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(
        output_dir / "metrics.csv",
        [
            {
                **cascade_totals,
                "precision": cascade_precision,
                "recall": cascade_recall,
                "macro_ap50": macro_ap50,
                "macro_map50_95": macro_map50_95,
                "detector_precision": detector_precision,
                "detector_recall": detector_recall,
                "matched_class_accuracy": matched_class_acc,
                "conf": args.conf,
                "ap_conf": args.ap_conf,
                "nms_iou": args.nms_iou,
                "match_iou": args.match_iou,
            }
        ],
    )
    report = {
        "detector_weights": args.detector_weights,
        "classifier_weights": args.classifier_weights,
        "cascade_totals": cascade_totals,
        "detector_totals": detector_totals,
        "precision": cascade_precision,
        "recall": cascade_recall,
        "macro_ap50": macro_ap50,
        "macro_map50_95": macro_map50_95,
        "detector_precision": detector_precision,
        "detector_recall": detector_recall,
        "matched_class_accuracy": matched_class_acc,
    }
    write_json(output_dir / "pipeline_report.json", report)
    print(f"级联评估完成: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估检测模型 + 分类模型级联效果。")
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--classifier-weights", required=True)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--subset", choices=["train", "val", "all"], default="val")
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "pipeline_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--ap-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
