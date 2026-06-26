from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .classifier_train import build_model, build_transforms, compute_metrics, plot_confusion_matrix, write_confusion_matrix
from .paths import DERIVED_DIR, RUNS_DIR


def evaluate(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.weights, map_location=args.device)
    model = build_model(checkpoint["model_name"], int(checkpoint["num_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    _, val_tf = build_transforms(int(checkpoint.get("input_size", args.image_size)), augment=False)
    val_ds = datasets.ImageFolder(dataset_dir / args.subset, transform=val_tf)
    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    confusion = np.zeros((int(checkpoint["num_classes"]), int(checkpoint["num_classes"])), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(args.device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            for gt, pred, prob in zip(targets.numpy(), preds, probs):
                confusion[int(gt), int(pred)] += 1
                rows.append({"gt": int(gt), "pred": int(pred), "score": float(prob[int(pred)])})
    class_names = checkpoint.get("class_names") or val_ds.classes
    metrics = compute_metrics(confusion)
    write_confusion_matrix(output_dir / "confusion_matrix.csv", confusion, class_names)
    plot_confusion_matrix(output_dir / "confusion_matrix.png", confusion)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "classification_eval.json").write_text(
        json.dumps({"metrics": metrics, "class_names": class_names, "dataset": str(dataset_dir)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"分类评估完成: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估离线 crop 分类模型。")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default=str(DERIVED_DIR / "classification_crops"))
    parser.add_argument("--subset", choices=["train", "val"], default="val")
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "classifier_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

