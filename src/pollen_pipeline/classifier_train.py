from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from .paths import DERIVED_DIR, RUNS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = DERIVED_DIR / "classification_crops"
SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "efficientnet_b0",
    "efficientnet_b1",
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_names(dataset_dir: Path, class_to_idx: dict[str, int]) -> list[str]:
    classes_path = dataset_dir / "classes.json"
    names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    if not classes_path.exists():
        return names
    payload = json.loads(classes_path.read_text(encoding="utf-8"))
    by_folder = {entry["folder"]: str(entry["name"]) for entry in payload.get("classes", [])}
    return [by_folder.get(folder, folder) for folder, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def build_transforms(image_size: int, augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[Any] = []
    if augment:
        train_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02)],
                    p=0.7,
                ),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    shear=0,
                    fill=0,
                ),
            ]
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_steps = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(train_steps), transforms.Compose(val_steps)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    normalized = model_name.lower()
    if normalized == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "wide_resnet50_2":
        weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        model = models.wide_resnet50_2(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "wide_resnet101_2":
        weights = models.Wide_ResNet101_2_Weights.DEFAULT if pretrained else None
        model = models.wide_resnet101_2(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if normalized == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classifier model: {model_name}")


def resolve_device(device_arg: str) -> torch.device:
    normalized = str(device_arg).strip().lower()
    if normalized.isdigit():
        return torch.device(f"cuda:{normalized}")
    if normalized in {"cuda", "cpu", "mps"}:
        return torch.device(normalized)
    if normalized.startswith(("cuda:", "cpu:", "mps:")):
        return torch.device(normalized)
    return torch.device(device_arg)


def cap_num_workers(requested: int) -> int:
    requested = max(0, int(requested))
    if requested == 0:
        return 0
    cpu_total = os.cpu_count() or requested
    safe_max = max(1, min(cpu_total, 2))
    return min(requested, safe_max)


def class_weights_from_targets(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (float(num_classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(confusion: np.ndarray) -> dict[str, Any]:
    total = float(confusion.sum())
    correct = float(np.trace(confusion))
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for index in range(confusion.shape[0]):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        p = tp / max(tp + fp, 1.0)
        r = tp / max(tp + fn, 1.0)
        precision.append(p)
        recall.append(r)
        f1.append(2.0 * p * r / max(p + r, 1e-12))
    return {
        "accuracy": correct / max(total, 1.0),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_class": [
            {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(confusion[i, :].sum())}
            for i in range(confusion.shape[0])
        ],
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    num_classes = len(loader.dataset.classes)  # type: ignore[attr-defined]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_seen = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = int(targets.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_seen += batch_size
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        truth = targets.detach().cpu().numpy()
        for gt, pred in zip(truth, preds):
            confusion[int(gt), int(pred)] += 1

    return total_loss / max(total_seen, 1), confusion


def write_confusion_matrix(path: Path, confusion: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt\\pred", *[f"class_{i}" for i in range(len(class_names))]])
        for index, row in enumerate(confusion.tolist()):
            writer.writerow([f"class_{index}", *row])


def plot_confusion_matrix(path: Path, confusion: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(confusion.shape[1]))
    ax.set_yticks(range(confusion.shape[0]))
    for y in range(confusion.shape[0]):
        for x in range(confusion.shape[1]):
            ax.text(x, y, str(int(confusion[y, x])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    class_names: list[str],
    class_to_idx: dict[str, int],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "num_classes": len(class_names),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "input_size": args.image_size,
            "metrics": metrics,
            "preprocessing": {
                "source": "fused.jpg crop",
                "crop": "bbox width and height scaled independently, then black-padded letterbox",
                "crop_scale": args.crop_scale,
                "no_stretch": True,
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    run_dir = Path(args.runs_dir).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=args.exist_ok)
    train_tf, val_tf = build_transforms(args.image_size, args.augment)
    train_ds = datasets.ImageFolder(dataset_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(dataset_dir / "val", transform=val_tf)
    num_classes = len(train_ds.classes)
    class_names = load_class_names(dataset_dir, train_ds.class_to_idx)

    weights = class_weights_from_targets(train_ds.targets, num_classes)
    sample_weights = [float(weights[target]) for target in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    worker_count = cap_num_workers(args.workers)
    if worker_count != args.workers:
        print(f"workers capped from {args.workers} to {worker_count} for this environment")
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=worker_count)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=worker_count)

    device = resolve_device(args.device)
    model = build_model(args.model, num_classes, args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)

    results_path = run_dir / "results.csv"
    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    best_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_confusion = run_epoch(model, val_loader, criterion, device, None)
        metrics = compute_metrics(val_confusion)
        scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "time": round(time.time() - start, 4),
            "train/loss": round(train_loss, 6),
            "val/loss": round(val_loss, 6),
            "metrics/accuracy": round(metrics["accuracy"], 6),
            "metrics/macro_precision": round(metrics["macro_precision"], 6),
            "metrics/macro_recall": round(metrics["macro_recall"], 6),
            "metrics/macro_f1": round(metrics["macro_f1"], 6),
            "metrics/balanced_accuracy": round(metrics["balanced_accuracy"], 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for index, per_class in enumerate(metrics["per_class"]):
            row[f"class_{index}/precision"] = round(per_class["precision"], 6)
            row[f"class_{index}/recall"] = round(per_class["recall"], 6)
            row[f"class_{index}/f1"] = round(per_class["f1"], 6)
            row[f"class_{index}/support"] = per_class["support"]
        rows.append(row)

        with results_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        save_checkpoint(run_dir / "weights" / "last.pt", model, args, class_names, train_ds.class_to_idx, metrics)
        if metrics["macro_f1"] > best_score:
            best_score = float(metrics["macro_f1"])
            best_metrics = metrics
            best_confusion = val_confusion.copy()
            save_checkpoint(run_dir / "weights" / "best.pt", model, args, class_names, train_ds.class_to_idx, metrics)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )

    write_confusion_matrix(run_dir / "confusion_matrix.csv", best_confusion, class_names)
    plot_confusion_matrix(run_dir / "confusion_matrix.png", best_confusion)
    report = {
        "best_metric": "macro_f1",
        "best_macro_f1": best_score,
        "class_names": class_names,
        "best_metrics": best_metrics,
        "dataset": str(dataset_dir),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    (run_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "model_info.json").write_text(
        json.dumps(
            {
                "task": "pollen_second_stage_classification",
                "model": args.model,
                "input_size": args.image_size,
                "class_names": class_names,
                "preprocessing": {
                    "crop_scale": args.crop_scale,
                    "letterbox": True,
                    "fill": "black",
                    "no_stretch": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Classifier run written: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练离线裁剪后的花粉二级分类模型。")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--runs-dir", default=str(RUNS_DIR / "classifier"))
    parser.add_argument("--run-name", default="classify_fused_resnet18")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet18")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--pretrained", type=parse_bool, default=False)
    parser.add_argument("--augment", type=parse_bool, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
