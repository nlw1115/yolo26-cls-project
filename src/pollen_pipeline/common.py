from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_cache(value: str | bool) -> bool | str:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"disk", "磁盘"}:
        raise argparse.ArgumentTypeError("为保持数据集精简，不允许 cache=disk。请使用 false/true/ram。")
    if normalized == "ram":
        return "ram"
    return parse_bool(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

