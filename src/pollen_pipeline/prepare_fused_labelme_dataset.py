from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .paths import DATASET_DIR


def prepare(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    groups_dir = dataset_dir / "groups"
    fused_dir = dataset_dir / "fused_labelme"
    fused_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for group_dir in sorted(p for p in groups_dir.iterdir() if p.is_dir()):
        fused = group_dir / args.fused_name
        labelme = group_dir / args.labelme_name
        if not fused.exists() or not labelme.exists():
            print(f"[跳过] 缺少 fused 或 labelme: {group_dir}")
            continue
        stem = group_dir.name
        image_out = fused_dir / f"{stem}.jpg"
        json_out = fused_dir / f"{stem}.json"
        shutil.copy2(fused, image_out)
        payload = json.loads(labelme.read_text(encoding="utf-8"))
        # 训练目录中图片与 JSON 同名，后续 LabelMe 直接打开也更稳定。
        payload["imagePath"] = image_out.name
        payload["imageData"] = None
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied += 1
    print(f"已整理融合图 LabelMe 训练目录: {fused_dir}")
    print(f"样本数: {copied}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 dataset/groups 整理训练用 fused_labelme 目录。")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--fused-name", default="fused.jpg")
    parser.add_argument("--labelme-name", default="labelme.json")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())

