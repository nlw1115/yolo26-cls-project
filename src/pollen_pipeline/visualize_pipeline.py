from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from .classifier_train import build_model, build_transforms
from .crop import crop_letterbox_from_bbox, imread_unicode, imwrite_unicode
from .detection_utils import draw_box, predict_image
from .labelme_dataset import load_class_names, load_records
from .paths import DATASET_DIR, RUNS_DIR


def visualize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(Path(args.dataset_dir).resolve() / "classes.yaml")
    _, records, _ = load_records(dataset_dir=Path(args.dataset_dir), subset=args.subset, class_mode="multi")
    if args.limit > 0:
        records = records[: args.limit]
    detector = YOLO(str(Path(args.detector_weights).resolve()))
    checkpoint = torch.load(args.classifier_weights, map_location=args.device)
    classifier = build_model(checkpoint["model_name"], int(checkpoint["num_classes"]), pretrained=False)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(args.device).eval()
    _, transform = build_transforms(int(checkpoint.get("input_size", args.image_size)), augment=False)
    for record in records:
        image = imread_unicode(record.image_path)
        predictions = predict_image(detector, record.image_path, args.conf, args.nms_iou, args.imgsz, args.device, args.max_det)
        for pred in predictions:
            crop = crop_letterbox_from_bbox(image, [pred["x1"], pred["y1"], pred["x2"], pred["y2"]], args.crop_scale, args.image_size)
            rgb = crop[:, :, ::-1].copy()
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                probs = torch.softmax(classifier(tensor), dim=1)[0].detach().cpu().numpy()
            cls_id = int(probs.argmax())
            label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            draw_box(image, [pred["x1"], pred["y1"], pred["x2"], pred["y2"]], (255, 0, 0), f"{label} {float(probs[cls_id]):.2f}")
        cv2.putText(image, f"pred={len(predictions)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        imwrite_unicode(output_dir / f"{record.stem}.jpg", image)
    print(f"级联可视化输出: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测 + 分类级联可视化。")
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--classifier-weights", required=True)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--subset", choices=["train", "val", "all"], default="val")
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "pipeline_visualize"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())

