from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .crop import imread_unicode, imwrite_unicode
from .labelme_dataset import Box, Record


def iou_xyxy(a: list[float] | tuple[float, ...] | np.ndarray, b: list[float] | tuple[float, ...] | np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def predict_image(
    model: YOLO,
    image_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        verbose=False,
    )
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for index, (box, score, cls_id) in enumerate(zip(xyxy, confs, clss)):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        rows.append(
            {
                "pred_index": index,
                "pred_cls": int(cls_id),
                "conf": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return rows


def match_predictions(
    predictions: list[dict[str, Any]],
    gt_boxes: tuple[Box, ...],
    match_iou: float,
    class_aware: bool = False,
) -> tuple[list[dict[str, Any]], set[int]]:
    matched_gt: set[int] = set()
    rows: list[dict[str, Any]] = []
    for pred in sorted(predictions, key=lambda row: float(row["conf"]), reverse=True):
        best_iou = 0.0
        best_gt = -1
        pred_box = [pred["x1"], pred["y1"], pred["x2"], pred["y2"]]
        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            if class_aware and int(pred["pred_cls"]) != int(gt.cls):
                continue
            value = iou_xyxy(pred_box, gt.bbox_xyxy)
            if value > best_iou:
                best_iou = value
                best_gt = gt_index
        matched = best_gt >= 0 and best_iou >= match_iou
        if matched:
            matched_gt.add(best_gt)
        rows.append(
            {
                **pred,
                "matched": matched,
                "gt_index": best_gt if matched else "",
                "gt_cls": gt_boxes[best_gt].original_cls if matched else "",
                "gt_label": gt_boxes[best_gt].original_label if matched else "",
                "match_iou": best_iou if matched else 0.0,
            }
        )
    return rows, matched_gt


def draw_box(image: np.ndarray, box: list[float] | tuple[float, ...], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    if label:
        y_text = max(15, y1 - 4)
        cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def write_detection_overlay(
    record: Record,
    matched_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = imread_unicode(record.image_path)
    for gt_index, box in enumerate(record.boxes):
        draw_box(image, box.bbox_xyxy, (0, 180, 0), f"GT {gt_index}")
    for row in matched_rows:
        pred_box = [row["x1"], row["y1"], row["x2"], row["y2"]]
        if row["matched"]:
            draw_box(image, pred_box, (255, 0, 0), f"TP {row['conf']:.2f}")
        else:
            draw_box(image, pred_box, (0, 0, 255), f"FP {row['conf']:.2f}")
    fn = len(record.boxes) - sum(1 for row in matched_rows if row["matched"])
    fp = sum(1 for row in matched_rows if not row["matched"])
    cv2.putText(image, f"GT={len(record.boxes)} FP={fp} FN={fn}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"GT={len(record.boxes)} FP={fp} FN={fn}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    imwrite_unicode(output_path, image)

