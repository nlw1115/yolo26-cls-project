from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .labelme_dataset import load_class_names, load_records
from .paths import DATASET_DIR


PALETTE = {
    0: (32, 139, 80),
    1: (29, 100, 220),
    2: (204, 110, 0),
    3: (160, 62, 180),
}
TEXT_COLOR = (20, 20, 20)
LABEL_BG = (255, 255, 255)
FP_LABEL_BG = (170, 24, 24)
FP_LABEL_TEXT = (255, 255, 255)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def read_predictions(path: Path) -> dict[str, list[dict[str, str]]]:
    by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_stem[row["stem"]].append(row)
    for rows in by_stem.values():
        rows.sort(key=lambda row: float(row.get("conf") or 0.0), reverse=True)
    return by_stem


def text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: Any,
    bg_color: tuple[int, int, int] = LABEL_BG,
    text_color: tuple[int, int, int] | None = None,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg_color, outline=color, width=1)
    draw.text((x, y), text, fill=text_color or color, font=font)


def class_label(class_names: list[str], cls_id: int, score: float | None = None) -> str:
    name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
    if score is None:
        return name
    return f"{name} {score:.2f}"


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float] | tuple[float, ...],
    color: tuple[int, int, int],
    label: str,
    font: Any,
    width: int = 3,
    label_bg: tuple[int, int, int] = LABEL_BG,
    label_text: tuple[int, int, int] | None = None,
) -> None:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    text_box(draw, (max(0.0, x1), max(0.0, y1 - 20)), label, color, font, label_bg, label_text)


def draw_predictions(
    image: Image.Image,
    rows: list[dict[str, str]],
    class_names: list[str],
    title: str,
    final_match_key: str,
    font: Any,
    title_font: Any,
) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    for row in rows:
        cls_id = int(row["pred_cls"])
        score = float(row.get("conf") or 0.0)
        matched = str(row.get(final_match_key, row.get("matched", "False"))).lower() == "true"
        color = PALETTE.get(cls_id, TEXT_COLOR)
        label = class_label(class_names, cls_id, score)
        label_bg = LABEL_BG
        label_text = None
        if not matched:
            label = f"FP {label}"
            label_bg = FP_LABEL_BG
            label_text = FP_LABEL_TEXT
        draw_box(
            draw,
            [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
            color,
            label,
            font,
            width=3 if matched else 2,
            label_bg=label_bg,
            label_text=label_text,
        )
    text_box(draw, (8, 8), f"{title} pred={len(rows)}", TEXT_COLOR, title_font)
    return panel


def draw_gt(image: Image.Image, record: Any, class_names: list[str], font: Any, title_font: Any) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    for index, box in enumerate(record.boxes):
        cls_id = int(box.cls)
        color = PALETTE.get(cls_id, TEXT_COLOR)
        draw_box(draw, box.bbox_xyxy, color, f"GT {index} {class_label(class_names, cls_id)}", font)
    text_box(draw, (8, 8), f"GT target={len(record.boxes)}", TEXT_COLOR, title_font)
    return panel


def make_compare_canvas(panels: list[Image.Image]) -> Image.Image:
    width, height = panels[0].size
    gap = 12
    canvas = Image.new("RGB", (width * len(panels) + gap * (len(panels) - 1), height), (245, 245, 245))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += width + gap
    return canvas


def make_contact_sheet(rows: list[dict[str, Any]], output_path: Path, font: Any) -> None:
    tile_width = 960
    tile_height = 360
    sheet = Image.new("RGB", (tile_width, tile_height * len(rows)), (250, 250, 250))
    for index, row in enumerate(rows):
        image = Image.open(row["image"]).convert("RGB")
        image.thumbnail((tile_width, 320), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), (250, 250, 250))
        tile.paste(image, ((tile_width - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (8, 328),
            f"{row['stem']}  GT={row['gt']}  A={row['a_pred']}  B={row['b_pred']}",
            fill=TEXT_COLOR,
            font=font,
        )
        sheet.paste(tile, (0, index * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def visualize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(load_class_names(Path(args.dataset_dir).resolve() / "classes.yaml"))
    _, records, _ = load_records(dataset_dir=Path(args.dataset_dir), subset=args.subset, class_mode="multi")
    records = sorted(records, key=lambda record: record.stem)
    if args.limit > 0:
        records = records[: args.limit]

    pred_a = read_predictions(Path(args.detector_predictions_a).resolve())
    pred_b = read_predictions(Path(args.pipeline_predictions_b).resolve())
    font = load_font(args.font_size)
    title_font = load_font(args.title_font_size)
    summary_rows: list[dict[str, Any]] = []

    for record in records:
        image = Image.open(record.image_path).convert("RGB")
        rows_a = pred_a.get(record.stem, [])
        rows_b = pred_b.get(record.stem, [])
        canvas = make_compare_canvas(
            [
                draw_predictions(image, rows_a, class_names, "A 多类别检测", "matched", font, title_font),
                draw_predictions(image, rows_b, class_names, "B 检出+分类", "cascade_matched", font, title_font),
                draw_gt(image, record, class_names, font, title_font),
            ]
        )
        output_path = output_dir / f"{record.stem}_compare.jpg"
        canvas.save(output_path, quality=args.jpeg_quality)
        summary_rows.append(
            {
                "stem": record.stem,
                "gt": len(record.boxes),
                "a_pred": len(rows_a),
                "b_pred": len(rows_b),
                "image": str(output_path),
            }
        )

    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "gt", "a_pred", "b_pred", "image"])
        writer.writeheader()
        writer.writerows(summary_rows)
    make_contact_sheet(summary_rows, output_dir / "all_val_compare_contact_sheet.jpg", font)
    print(f"对比检测图输出: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把两种检测方案的预测结果绘制为并排对照图。")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--subset", choices=["train", "val", "all"], default="val")
    parser.add_argument("--detector-predictions-a", required=True, help="A 多类别检测 predictions.csv。")
    parser.add_argument("--pipeline-predictions-b", required=True, help="B 级联 pipeline_predictions.csv。")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--font-size", type=int, default=13)
    parser.add_argument("--title-font-size", type=int, default=20)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
