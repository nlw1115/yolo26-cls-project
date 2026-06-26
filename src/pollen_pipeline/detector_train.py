from __future__ import annotations

import argparse
import csv
import math
from copy import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOCAL_RANK, LOGGER, RANK
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first, unwrap_model

from .common import parse_bool, parse_cache, write_json
from .labelme_dataset import DatasetBundle, Record, load_bundle, path_key
from .paths import DATASET_DIR, PROJECT_ROOT, RUNS_DIR, WEIGHTS_DIR


@dataclass
class RuntimeContext:
    bundle: DatasetBundle
    data_repeat: int
    augment: bool
    class_balance_alpha: float
    class_balance_max_repeat: int
    class_loss_weights: list[float] | None


_CONTEXT: RuntimeContext | None = None


def set_context(context: RuntimeContext) -> None:
    global _CONTEXT
    _CONTEXT = context


def clear_context() -> None:
    global _CONTEXT
    _CONTEXT = None


def get_context() -> RuntimeContext:
    if _CONTEXT is None:
        raise RuntimeError("训练上下文尚未初始化")
    return _CONTEXT


class LabelMeYOLODataset(YOLODataset):
    """把 LabelMe JSON 直接适配成 Ultralytics YOLODataset，不生成 YOLO txt 标签。"""

    def __init__(
        self,
        *args: Any,
        record_by_image: dict[str, Record],
        repeat: int = 1,
        class_balance_alpha: float = 0.0,
        class_balance_max_repeat: int = 1,
        **kwargs: Any,
    ) -> None:
        self.record_by_image = record_by_image
        self.repeat = max(1, int(repeat))
        self.class_balance_alpha = max(0.0, float(class_balance_alpha))
        self.class_balance_max_repeat = max(1, int(class_balance_max_repeat))
        self.sample_indices: list[int] | None = None
        super().__init__(*args, **kwargs)
        self.sample_indices = self.build_sample_indices()

    def _base_index(self, index: int) -> int:
        if self.sample_indices:
            return self.sample_indices[index % len(self.sample_indices)]
        return index % self.ni if self.ni else index

    def __len__(self) -> int:
        if self.sample_indices is not None:
            return len(self.sample_indices)
        return self.ni * self.repeat

    def build_sample_indices(self) -> list[int]:
        if not self.im_files:
            return []
        if self.class_balance_alpha <= 0:
            return [index for index in range(self.ni) for _ in range(self.repeat)]

        class_counts: Counter[int] = Counter()
        image_classes: list[set[int]] = []
        for image_file in self.im_files:
            record = self.record_by_image[path_key(image_file)]
            classes = {int(box.cls) for box in record.boxes}
            image_classes.append(classes)
            for box in record.boxes:
                class_counts[int(box.cls)] += 1

        max_count = max(class_counts.values(), default=1)
        sample_indices: list[int] = []
        image_repeats: list[int] = []
        for index, classes in enumerate(image_classes):
            balance_factor = max(
                ((max_count / max(class_counts[cls], 1)) ** self.class_balance_alpha for cls in classes),
                default=1.0,
            )
            repeat_count = min(
                self.repeat * self.class_balance_max_repeat,
                max(1, int(math.ceil(self.repeat * balance_factor))),
            )
            image_repeats.append(repeat_count)
            sample_indices.extend([index] * repeat_count)

        print(
            f"{self.prefix}class-balanced sampling: base_images={self.ni} samples={len(sample_indices)} "
            f"repeat_range={min(image_repeats)}-{max(image_repeats)} alpha={self.class_balance_alpha} "
            f"max_repeat={self.class_balance_max_repeat}"
        )
        return sample_indices

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        return super().get_image_and_label(self._base_index(index))

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        if isinstance(img_path, (list, tuple)):
            image_files = [str(Path(p).resolve()) for p in img_path]
        else:
            path = Path(img_path)
            image_files = [str(path.resolve())] if path.is_file() else super().get_img_files(img_path)
        if not image_files:
            raise FileNotFoundError("没有可读取的训练图片")
        return image_files

    def get_labels(self) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []
        missing: list[str] = []
        total_boxes = 0
        for image_file in self.im_files:
            record = self.record_by_image.get(path_key(image_file))
            if record is None:
                missing.append(image_file)
                continue
            cls = np.array([[box.cls] for box in record.boxes], dtype=np.float32)
            bboxes = np.array([[box.xc, box.yc, box.w, box.h] for box in record.boxes], dtype=np.float32)
            total_boxes += len(record.boxes)
            labels.append(
                {
                    "im_file": str(record.image_path),
                    "shape": (record.height, record.width),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if missing:
            shown = "\n  ".join(missing[:10])
            raise RuntimeError(f"以下图片没有可用 LabelMe 标注:\n  {shown}")
        if not labels:
            raise RuntimeError("没有可用于训练的 LabelMe 标注")
        self.im_files = [label["im_file"] for label in labels]
        print(f"{self.prefix}直接读取 LabelMe: {len(labels)} 张图，{total_boxes} 个目标")
        return labels


class LabelMeDetectionValidator(DetectionValidator):
    def display_class_name(self, cls_id: int) -> str:
        try:
            class_names = get_context().bundle.class_names
            if 0 <= cls_id < len(class_names):
                return f"class_{cls_id}:{class_names[cls_id]}"
        except RuntimeError:
            pass
        try:
            name = self.names[cls_id]
        except (KeyError, IndexError, TypeError):
            name = ""
        name = str(name).strip()
        return f"class_{cls_id}:{name}" if name else f"class_{cls_id}"

    def print_results(self) -> None:
        """Print aggregate and per-class metrics during training validation."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")

        if self.args.verbose and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.display_class_name(int(c)),
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )


class LabelMeDetectionTrainer(DetectionTrainer):
    def get_dataset(self) -> dict[str, Any]:
        context = get_context()
        names = {idx: name for idx, name in enumerate(context.bundle.class_names)}
        return {
            "path": str(PROJECT_ROOT),
            "train": [str(record.image_path) for record in context.bundle.train_records],
            "val": [str(record.image_path) for record in context.bundle.val_records],
            "nc": len(names),
            "names": names,
            "channels": 3,
        }

    def get_model(self, cfg: str | None = None, weights: Any | None = None, verbose: bool = True) -> DetectionModel:
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        context = get_context()
        if context.class_loss_weights is not None:
            model.class_weights = torch.tensor(context.class_loss_weights, dtype=torch.float32)
        return model

    def build_dataset(self, img_path: str | list[str], mode: str = "train", batch: int | None = None):
        context = get_context()
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return LabelMeYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=context.augment and mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            record_by_image=context.bundle.record_by_image,
            repeat=context.data_repeat if mode == "train" else 1,
            class_balance_alpha=context.class_balance_alpha if mode == "train" else 0.0,
            class_balance_max_repeat=context.class_balance_max_repeat,
        )

    def get_dataloader(self, dataset_path: str | list[str], batch_size: int = 16, rank: int = 0, mode: str = "train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=mode == "train",
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return LabelMeDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def final_eval(self) -> None:
        """Use the LabelMe dataloader for final validation instead of resolving args.data as a YAML path."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})

        if not model:
            return

        LOGGER.info(f"\nValidating {model} with LabelMe direct reader...")
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        metrics, _ = self.validate()
        self.metrics = metrics or {}
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")

    def read_results_csv(self) -> dict[str, list[Any]]:
        if not self.csv.exists():
            return {}
        with self.csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {}
        result: dict[str, list[Any]] = {key: [] for key in rows[0].keys()}
        for row in rows:
            for key, value in row.items():
                try:
                    result[key].append(float(value))
                except (TypeError, ValueError):
                    result[key].append(value)
        return result


def resolve_pretrained(model: str, pretrained: str) -> str | bool:
    if pretrained.lower() in {"false", "0", "none", "no"}:
        return False
    if pretrained.lower() != "auto":
        return pretrained
    model_name = Path(model).stem
    local = WEIGHTS_DIR / f"{model_name}.pt"
    return str(local) if local.exists() else f"{model_name}.pt"


def compute_class_loss_weights(bundle: DatasetBundle, alpha: float, max_weight: float) -> list[float] | None:
    if alpha <= 0:
        return None
    counts = Counter(int(box.cls) for record in bundle.train_records for box in record.boxes)
    if not counts:
        return None
    max_count = max(counts.values())
    weights = np.ones(len(bundle.class_names), dtype=np.float32)
    for cls_id in range(len(weights)):
        count = max(counts.get(cls_id, 0), 1)
        weights[cls_id] = float((max_count / count) ** alpha)
    weights /= max(float(weights.mean()), 1e-12)
    if max_weight > 0:
        weights = np.minimum(weights, float(max_weight))
        weights /= max(float(weights.mean()), 1e-12)
    return [float(value) for value in weights.tolist()]


def train(args: argparse.Namespace) -> None:
    bundle = load_bundle(
        dataset_dir=Path(args.dataset_dir),
        class_mode=args.class_mode,
        single_class_name=args.single_class_name,
        min_box_size_px=args.min_box_size_px,
    )
    print("数据读取摘要:")
    print(bundle.summary)
    class_loss_weights = compute_class_loss_weights(bundle, args.class_weight_alpha, args.class_weight_max)
    if class_loss_weights is not None:
        shown = ", ".join(
            f"class_{idx}:{name}={weight:.3f}"
            for idx, (name, weight) in enumerate(zip(bundle.class_names, class_loss_weights))
        )
        print(f"class loss weights: {shown}")
    set_context(
        RuntimeContext(
            bundle=bundle,
            data_repeat=args.data_repeat,
            augment=args.augment,
            class_balance_alpha=args.class_balance_alpha,
            class_balance_max_repeat=args.class_balance_max_repeat,
            class_loss_weights=class_loss_weights,
        )
    )
    try:
        model = YOLO(args.model)
        pretrained = resolve_pretrained(args.model, args.pretrained)
        model.train(
            trainer=LabelMeDetectionTrainer,
            data="labelme_direct_read",
            project=str(Path(args.runs_dir).resolve()),
            name=args.run_name,
            exist_ok=args.exist_ok,
            pretrained=pretrained,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            warmup_bias_lr=args.warmup_bias_lr,
            box=args.box,
            cls=args.cls,
            dfl=args.dfl,
            freeze=args.freeze,
            patience=args.patience,
            amp=args.amp,
            cache=args.cache,
            seed=args.seed,
            augment=args.augment,
            mosaic=args.mosaic,
            mixup=args.mixup,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            plots=args.plots,
            verbose=args.show_class_metrics,
            task="detect",
        )
        run_dir = Path(args.runs_dir).resolve() / args.run_name
        write_json(
            run_dir / "labelme_train_meta.json",
            {
                "class_mode": args.class_mode,
                "class_names": list(bundle.class_names),
                "dataset_dir": str(Path(args.dataset_dir).resolve()),
                "data_repeat": args.data_repeat,
                "class_balance_alpha": args.class_balance_alpha,
                "class_balance_max_repeat": args.class_balance_max_repeat,
                "class_weight_alpha": args.class_weight_alpha,
                "class_weight_max": args.class_weight_max,
                "class_loss_weights": class_loss_weights,
                "summary": bundle.summary,
                "pretrained": pretrained,
            },
        )
    finally:
        clear_context()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="直接读取 LabelMe JSON 训练 YOLO26 检测模型。")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="数据集根目录，内部需要 fused_labelme/classes.yaml/split.json。")
    parser.add_argument("--class-mode", choices=["single", "multi"], default="single", help="single=训练时内存合并为 pollen；multi=保留原始多类别。")
    parser.add_argument("--single-class-name", default="pollen", help="单类别训练时写入模型的类别名。")
    parser.add_argument("--min-box-size-px", type=float, default=2.0)
    parser.add_argument("--model", default="yolo26s.yaml")
    parser.add_argument("--pretrained", default="auto", help="auto 会优先使用 weights/<model>.pt。")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR / "detector"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--data-repeat", type=int, default=4)
    parser.add_argument("--class-balance-alpha", type=float, default=0.0)
    parser.add_argument("--class-balance-max-repeat", type=int, default=4)
    parser.add_argument("--class-weight-alpha", type=float, default=0.0)
    parser.add_argument("--class-weight-max", type=float, default=3.0)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.02)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.01)
    parser.add_argument("--box", type=float, default=10.0)
    parser.add_argument("--cls", type=float, default=0.25)
    parser.add_argument("--dfl", type=float, default=2.0)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", type=parse_bool, default=False)
    parser.add_argument("--cache", type=parse_cache, default=False)
    parser.add_argument("--seed", type=int, default=26052016)
    parser.add_argument("--augment", type=parse_bool, default=True)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.03)
    parser.add_argument("--hsv-v", type=float, default=0.05)
    parser.add_argument("--plots", type=parse_bool, default=False)
    parser.add_argument("--show-class-metrics", type=parse_bool, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
