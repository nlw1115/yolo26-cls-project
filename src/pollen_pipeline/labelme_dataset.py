from __future__ import annotations

import base64
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from .paths import DATASET_DIR


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Box:
    cls: int
    label: str
    original_cls: int
    original_label: str
    bbox_xyxy: tuple[float, float, float, float]
    xc: float
    yc: float
    w: float
    h: float


@dataclass(frozen=True)
class Record:
    stem: str
    image_path: Path
    json_path: Path
    width: int
    height: int
    boxes: tuple[Box, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(box.original_label for box in self.boxes)


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    class_names: tuple[str, ...]
    train_records: tuple[Record, ...]
    val_records: tuple[Record, ...]
    record_by_image: dict[str, Record]
    summary: dict[str, Any]


def path_key(path: str | Path) -> str:
    return str(Path(path).resolve()).lower()


def load_class_names(path: Path) -> tuple[str, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = payload.get("names", payload) if isinstance(payload, dict) else payload
    if not isinstance(names, list) or not names:
        raise ValueError(f"类别文件无效: {path}")
    return tuple(str(name).strip() for name in names)


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = [str(value) for value in payload.get("train", [])]
    val = [str(value) for value in payload.get("val", [])]
    if not train or not val:
        raise ValueError(f"split.json 必须包含非空 train/val: {path}")
    return {"train": train, "val": val}


def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.glob(pattern))


def resolve_image_path(json_path: Path, payload: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    image_path = payload.get("imagePath")
    if image_path:
        raw = Path(str(image_path))
        candidates.append(raw if raw.is_absolute() else json_path.parent / raw)
    for ext in IMAGE_EXTS:
        candidates.append(json_path.with_suffix(ext))
        candidates.append(json_path.with_suffix(ext.upper()))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到 LabelMe JSON 对应图片: {json_path}")


def image_size(image_path: Path, payload: dict[str, Any]) -> tuple[int, int]:
    width = int(payload.get("imageWidth") or 0)
    height = int(payload.get("imageHeight") or 0)
    if width > 0 and height > 0:
        return width, height
    with Image.open(image_path) as img:
        return img.size


def points_bbox(points: list[list[float]]) -> tuple[float, float, float, float] | None:
    coords = [(float(p[0]), float(p[1])) for p in points if len(p) >= 2]
    if not coords:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return min(xs), min(ys), max(xs), max(ys)


def mask_bbox(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    mask_data = shape.get("mask")
    points = shape.get("points") or []
    if not mask_data or len(points) < 1:
        return None
    try:
        raw = base64.b64decode(mask_data)
        with Image.open(io.BytesIO(raw)) as mask:
            arr = np.asarray(mask.convert("L"))
    except Exception:
        return None
    ys, xs = np.nonzero(arr)
    if len(xs) == 0 or len(ys) == 0:
        return None
    origin_x = float(points[0][0])
    origin_y = float(points[0][1])
    return (
        origin_x + float(xs.min()),
        origin_y + float(ys.min()),
        origin_x + float(xs.max() + 1),
        origin_y + float(ys.max() + 1),
    )


def clip_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_box(
    cls_id: int,
    label: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    min_box_size_px: float,
    class_mode: str,
    single_class_name: str,
) -> Box | None:
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_box_size_px or box_h < min_box_size_px:
        return None
    mapped_cls = 0 if class_mode == "single" else cls_id
    mapped_label = single_class_name if class_mode == "single" else label
    return Box(
        cls=mapped_cls,
        label=mapped_label,
        original_cls=cls_id,
        original_label=label,
        bbox_xyxy=clipped,
        xc=((x1 + x2) / 2.0) / float(width),
        yc=((y1 + y2) / 2.0) / float(height),
        w=box_w / float(width),
        h=box_h / float(height),
    )


def parse_labelme_json(
    json_path: Path,
    class_names: tuple[str, ...],
    class_mode: str = "multi",
    single_class_name: str = "pollen",
    min_box_size_px: float = 2.0,
    use_mask_bbox: bool = True,
) -> Record | None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = resolve_image_path(json_path, payload)
    width, height = image_size(image_path, payload)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    boxes: list[Box] = []
    unknown_labels: set[str] = set()
    for shape in payload.get("shapes") or []:
        label = str(shape.get("label") or "").strip()
        if label not in class_to_id:
            unknown_labels.add(label)
            continue
        # LabelMe 的 mask 比两点外接框更贴近实例；没有 mask 时退回 points 外接框。
        bbox = mask_bbox(shape) if use_mask_bbox else None
        if bbox is None:
            bbox = points_bbox(shape.get("points") or [])
        if bbox is None:
            continue
        box = build_box(
            cls_id=class_to_id[label],
            label=label,
            bbox=bbox,
            width=width,
            height=height,
            min_box_size_px=min_box_size_px,
            class_mode=class_mode,
            single_class_name=single_class_name,
        )
        if box is not None:
            boxes.append(box)
    if unknown_labels:
        names = ", ".join(sorted(unknown_labels))
        raise ValueError(f"标注文件出现未配置类别: {json_path}; 类别={names}")
    if not boxes:
        return None
    return Record(
        stem=json_path.stem,
        image_path=image_path,
        json_path=json_path,
        width=width,
        height=height,
        boxes=tuple(boxes),
    )


def load_records(
    dataset_dir: Path = DATASET_DIR,
    subset: str = "all",
    class_mode: str = "multi",
    single_class_name: str = "pollen",
    min_box_size_px: float = 2.0,
) -> tuple[tuple[str, ...], list[Record], dict[str, Any]]:
    dataset_dir = Path(dataset_dir).resolve()
    class_names = load_class_names(dataset_dir / "classes.yaml")
    labelme_dir = dataset_dir / "fused_labelme"
    split = load_split(dataset_dir / "split.json")
    allowed: set[str] | None = None
    if subset != "all":
        if subset not in split:
            raise ValueError(f"subset 必须是 train/val/all，当前: {subset}")
        allowed = set(split[subset])
    records: list[Record] = []
    label_counts: Counter[str] = Counter()
    for json_path in iter_files(labelme_dir, "*.json"):
        if allowed is not None and json_path.stem not in allowed:
            continue
        record = parse_labelme_json(
            json_path=json_path,
            class_names=class_names,
            class_mode=class_mode,
            single_class_name=single_class_name,
            min_box_size_px=min_box_size_px,
        )
        if record is None:
            continue
        records.append(record)
        label_counts.update(record.labels)
    summary = {
        "dataset_dir": str(dataset_dir),
        "subset": subset,
        "class_mode": class_mode,
        "image_count": len(records),
        "box_count": sum(len(record.boxes) for record in records),
        "label_counts": dict(sorted(label_counts.items())),
    }
    if class_mode == "single":
        return (single_class_name,), records, summary
    return class_names, records, summary


def load_bundle(
    dataset_dir: Path = DATASET_DIR,
    class_mode: str = "multi",
    single_class_name: str = "pollen",
    min_box_size_px: float = 2.0,
) -> DatasetBundle:
    dataset_dir = Path(dataset_dir).resolve()
    class_names, train_records, train_summary = load_records(
        dataset_dir=dataset_dir,
        subset="train",
        class_mode=class_mode,
        single_class_name=single_class_name,
        min_box_size_px=min_box_size_px,
    )
    _, val_records, val_summary = load_records(
        dataset_dir=dataset_dir,
        subset="val",
        class_mode=class_mode,
        single_class_name=single_class_name,
        min_box_size_px=min_box_size_px,
    )
    record_by_image = {
        path_key(record.image_path): record for record in [*train_records, *val_records]
    }
    return DatasetBundle(
        dataset_dir=dataset_dir,
        class_names=class_names,
        train_records=tuple(train_records),
        val_records=tuple(val_records),
        record_by_image=record_by_image,
        summary={"train": train_summary, "val": val_summary},
    )

