from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from .common import write_csv
from .detection_utils import iou_xyxy, match_predictions, predict_image, write_detection_overlay
from .labelme_dataset import load_records
from .paths import DATASET_DIR, RUNS_DIR


IOU_THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.5, 0.96, 0.05))


def class_name(class_names: tuple[str, ...], cls_id: int) -> str:
    return class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"


def round_metric(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0 or precision.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def compute_ap_by_class(
    records: tuple[Any, ...],
    predictions_by_stem: dict[str, list[dict[str, Any]]],
    class_ids: list[int],
) -> dict[int, dict[str, Any]]:
    gt_by_class_image: dict[int, dict[str, list[tuple[float, float, float, float]]]] = {
        cls_id: {} for cls_id in class_ids
    }
    pred_by_class: dict[int, list[dict[str, Any]]] = {cls_id: [] for cls_id in class_ids}

    for record in records:
        for box in record.boxes:
            cls_id = int(box.cls)
            gt_by_class_image.setdefault(cls_id, {}).setdefault(record.stem, []).append(box.bbox_xyxy)
        for pred in predictions_by_stem.get(record.stem, []):
            cls_id = int(pred["pred_cls"])
            pred_by_class.setdefault(cls_id, []).append(pred)

    result: dict[int, dict[str, Any]] = {}
    for cls_id in sorted(set(class_ids) | set(gt_by_class_image) | set(pred_by_class)):
        gt_by_image = gt_by_class_image.get(cls_id, {})
        preds = sorted(pred_by_class.get(cls_id, []), key=lambda row: float(row["conf"]), reverse=True)
        gt_total = sum(len(boxes) for boxes in gt_by_image.values())
        if gt_total == 0:
            result[cls_id] = {"ap50": None, "map50_95": None, "aps": {}}
            continue

        aps: dict[float, float] = {}
        for threshold in IOU_THRESHOLDS:
            matched: dict[str, set[int]] = {stem: set() for stem in gt_by_image}
            tp_values: list[float] = []
            fp_values: list[float] = []
            for pred in preds:
                stem = str(pred["stem"])
                gt_boxes = gt_by_image.get(stem, [])
                best_iou = 0.0
                best_gt = -1
                pred_box = [pred["x1"], pred["y1"], pred["x2"], pred["y2"]]
                for gt_index, gt_box in enumerate(gt_boxes):
                    if gt_index in matched.setdefault(stem, set()):
                        continue
                    value = iou_xyxy(pred_box, gt_box)
                    if value > best_iou:
                        best_iou = value
                        best_gt = gt_index
                if best_gt >= 0 and best_iou >= threshold:
                    matched[stem].add(best_gt)
                    tp_values.append(1.0)
                    fp_values.append(0.0)
                else:
                    tp_values.append(0.0)
                    fp_values.append(1.0)

            if not preds:
                aps[threshold] = 0.0
                continue
            tp_cum = np.cumsum(np.asarray(tp_values, dtype=np.float64))
            fp_cum = np.cumsum(np.asarray(fp_values, dtype=np.float64))
            recall = tp_cum / max(float(gt_total), 1.0)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            aps[threshold] = compute_ap(recall, precision)

        result[cls_id] = {
            "ap50": aps.get(0.5, 0.0),
            "map50_95": float(np.mean(list(aps.values()))),
            "aps": aps,
        }
    return result


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names, records, summary = load_records(
        dataset_dir=Path(args.dataset_dir),
        subset=args.subset,
        class_mode=args.class_mode,
        single_class_name=args.single_class_name,
    )
    model = YOLO(str(Path(args.weights).resolve()))
    per_image: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    predictions_for_ap: dict[str, list[dict[str, Any]]] = {}
    per_class_totals: dict[int, dict[str, int]] = {
        cls_id: {"gt": 0, "pred": 0, "tp": 0} for cls_id in range(len(class_names))
    }
    totals = {"images": 0, "gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}

    for record in records:
        raw_predictions = predict_image(
            model=model,
            image_path=record.image_path,
            conf=min(args.conf, args.ap_conf),
            iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )
        predictions = [row for row in raw_predictions if float(row["conf"]) >= args.conf]
        predictions_for_ap[record.stem] = [
            {"stem": record.stem, **row} for row in raw_predictions if float(row["conf"]) >= args.ap_conf
        ]
        matched_rows, matched_gt = match_predictions(
            predictions=predictions,
            gt_boxes=record.boxes,
            match_iou=args.match_iou,
            class_aware=args.class_aware,
        )
        tp = len(matched_gt)
        fp = len(predictions) - tp
        fn = len(record.boxes) - tp
        matched_ious = [row["match_iou"] for row in matched_rows if row["matched"]]
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        overlay_path = overlay_dir / f"{record.stem}.jpg"
        write_detection_overlay(record, matched_rows, overlay_path)

        for box in record.boxes:
            per_class_totals.setdefault(int(box.cls), {"gt": 0, "pred": 0, "tp": 0})["gt"] += 1
        for pred in predictions:
            per_class_totals.setdefault(int(pred["pred_cls"]), {"gt": 0, "pred": 0, "tp": 0})["pred"] += 1
        for row in matched_rows:
            if not row["matched"] or row["gt_cls"] == "":
                continue
            gt_cls = int(row["gt_cls"])
            pred_cls = int(row["pred_cls"])
            if pred_cls == gt_cls:
                per_class_totals.setdefault(gt_cls, {"gt": 0, "pred": 0, "tp": 0})["tp"] += 1

        per_image.append(
            {
                "stem": record.stem,
                "gt": len(record.boxes),
                "pred": len(predictions),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "mean_iou": mean_iou,
                "overlay": overlay_path.relative_to(output_dir).as_posix(),
            }
        )
        for row in matched_rows:
            pred_rows.append({"stem": record.stem, **row})
        totals["images"] += 1
        totals["gt"] += len(record.boxes)
        totals["pred"] += len(predictions)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
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
    macro_precision = float(np.mean([float(row["precision"]) for row in valid_class_rows])) if valid_class_rows else 0.0
    macro_recall = float(np.mean([float(row["recall"]) for row in valid_class_rows])) if valid_class_rows else 0.0
    macro_f1 = float(np.mean([float(row["f1"]) for row in valid_class_rows])) if valid_class_rows else 0.0
    valid_ap50 = [float(row["ap50"]) for row in valid_class_rows if row["ap50"] != ""]
    valid_map = [float(row["map50_95"]) for row in valid_class_rows if row["map50_95"] != ""]
    macro_ap50 = float(np.mean(valid_ap50)) if valid_ap50 else 0.0
    macro_map50_95 = float(np.mean(valid_map)) if valid_map else 0.0

    write_csv(output_dir / "per_image_metrics.csv", per_image)
    write_csv(output_dir / "predictions.csv", pred_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(
        output_dir / "metrics.csv",
        [
            {
                **totals,
                "precision": precision,
                "recall": recall,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "macro_ap50": macro_ap50,
                "macro_map50_95": macro_map50_95,
                "class_mode": args.class_mode,
                "class_aware": args.class_aware,
                "conf": args.conf,
                "ap_conf": args.ap_conf,
                "nms_iou": args.nms_iou,
                "match_iou": args.match_iou,
            }
        ],
    )

    worst_rows = sorted(per_image, key=lambda row: (row["fn"] + row["fp"], row["fn"]), reverse=True)[:20]
    report = [
        "# Detection evaluation report",
        "",
        f"- weights: `{args.weights}`",
        f"- subset: `{args.subset}`",
        f"- class mode: `{args.class_mode}`",
        f"- class aware: `{args.class_aware}`",
        f"- classes: `{list(class_names)}`",
        f"- conf / ap_conf / nms_iou / match_iou: `{args.conf}` / `{args.ap_conf}` / `{args.nms_iou}` / `{args.match_iou}`",
        f"- images: {totals['images']}",
        f"- GT / pred: {totals['gt']} / {totals['pred']}",
        f"- TP / FP / FN: {totals['tp']} / {totals['fp']} / {totals['fn']}",
        f"- precision / recall: {precision:.4f} / {recall:.4f}",
        f"- macro precision / recall / F1: {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}",
        f"- macro AP50 / mAP50-95: {macro_ap50:.4f} / {macro_map50_95:.4f}",
        "",
        "## Per-class metrics",
        "",
        "| class | gt | pred | tp | fp | fn | precision | recall | f1 | AP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_class_rows:
        report.append(
            f"| {row['class_name']} | {row['gt']} | {row['pred']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']} | {row['recall']} | {row['f1']} | {row['ap50']} | {row['map50_95']} |"
        )
    report.extend(
        [
            "",
            "## Overlay colors",
            "",
            "- Green: LabelMe ground truth box",
            "- Blue: matched true positive prediction",
            "- Red: false positive prediction",
            "- False negatives appear as green boxes without a matched blue box",
            "",
            "## Review samples",
            "",
            "| sample | gt | pred | tp | fp | fn | overlay |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in worst_rows:
        report.append(
            f"| {row['stem']} | {row['gt']} | {row['pred']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"[overlay]({row['overlay']}) |"
        )
    (output_dir / "detection_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Detection eval written: {output_dir}")
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector with LabelMe annotations.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--subset", choices=["train", "val", "all"], default="val")
    parser.add_argument("--class-mode", choices=["single", "multi"], default="single")
    parser.add_argument("--single-class-name", default="pollen")
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "detector_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--ap-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--class-aware", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
