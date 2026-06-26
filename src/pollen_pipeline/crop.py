from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    params: list[int] = []
    if suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode image for: {path}")
    encoded.tofile(str(path))


def expand_bbox_xyxy(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    crop_scale: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w = width * float(crop_scale)
    crop_h = height * float(crop_scale)
    return (
        cx - crop_w / 2.0,
        cy - crop_h / 2.0,
        cx + crop_w / 2.0,
        cy + crop_h / 2.0,
    )


def crop_with_black_padding(
    image: np.ndarray,
    crop_xyxy: tuple[float, float, float, float],
) -> np.ndarray:
    x1, y1, x2, y2 = crop_xyxy
    left = math.floor(x1)
    top = math.floor(y1)
    right = math.ceil(x2)
    bottom = math.ceil(y2)
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)

    if image.ndim == 2:
        output = np.zeros((crop_h, crop_w), dtype=image.dtype)
    else:
        output = np.zeros((crop_h, crop_w, image.shape[2]), dtype=image.dtype)

    src_h, src_w = image.shape[:2]
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(src_w, right)
    src_bottom = min(src_h, bottom)
    if src_right <= src_left or src_bottom <= src_top:
        return output

    dst_left = src_left - left
    dst_top = src_top - top
    dst_right = dst_left + (src_right - src_left)
    dst_bottom = dst_top + (src_bottom - src_top)
    output[dst_top:dst_bottom, dst_left:dst_right] = image[src_top:src_bottom, src_left:src_right]
    return output


def letterbox_to_square(
    image: np.ndarray,
    image_size: int = 224,
    fill_value: int = 0,
) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Cannot letterbox an empty image")
    scale = min(float(image_size) / float(w), float(image_size) / float(h))
    new_w = max(1, min(image_size, int(round(w * scale))))
    new_h = max(1, min(image_size, int(round(h * scale))))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        canvas = np.full((image_size, image_size), fill_value, dtype=image.dtype)
    else:
        canvas = np.full((image_size, image_size, image.shape[2]), fill_value, dtype=image.dtype)

    x0 = (image_size - new_w) // 2
    y0 = (image_size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def crop_letterbox_from_bbox(
    image: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    crop_scale: float = 1.6,
    image_size: int = 224,
) -> np.ndarray:
    crop_xyxy = expand_bbox_xyxy(bbox_xyxy, crop_scale)
    crop = crop_with_black_padding(image, crop_xyxy)
    return letterbox_to_square(crop, image_size=image_size, fill_value=0)
