from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np
from PIL import Image
from PIL import ImageOps


@dataclass(frozen=True)
class PromptPoint:
    label: str
    x: float
    y: float
    negative_points: tuple[tuple[float, float], ...] = ()
    group_id: int | None = None
    description: str = ""
    flags: dict[str, bool] | None = None


def generate_labelme_json_from_sam2_points(
    image_path: str | Path,
    prompt_points: Iterable[PromptPoint | dict[str, Any]],
    output_path: str | Path | None = None,
    model_name: str = "sam2:tiny",
    overwrite: bool = False,
    image_data: bool = False,
    min_mask_area: int = 1,
) -> dict[str, Any]:
    """Generate a LabelMe mask JSON from one SAM-style point per object.

    Each prompt is evaluated independently. This matters because multiple
    foreground points in one SAM/SAM2 call mean "one object with multiple
    positive hints", not "multiple objects".
    """

    image_path = Path(image_path).resolve()
    prompts = [coerce_prompt_point(p) for p in prompt_points]
    if not prompts:
        raise ValueError("prompt_points must contain at least one point")

    output: Path | None = None
    if output_path is not None:
        output = Path(output_path).resolve()
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output JSON already exists: {output}")

    image = load_rgb_image(image_path)
    labeler = Sam2PointLabeler(model_name=model_name)

    shapes: list[dict[str, Any]] = []
    image_id = str(image_path)
    for prompt in prompts:
        shape = labeler.shape_from_prompt(
            image=image,
            image_id=image_id,
            prompt=prompt,
            min_mask_area=min_mask_area,
        )
        if shape is not None:
            shapes.append(shape)

    height, width = image.shape[:2]
    payload: dict[str, Any] = {
        "version": get_labelme_version(),
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name
        if output is None
        else make_labelme_image_path(image_path=image_path, label_path=output),
        "imageData": encode_image_file(image_path) if image_data else None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


class Sam2PointLabeler:
    def __init__(self, model_name: str = "sam2:tiny") -> None:
        patch_osam_blob_path_for_windows()
        from labelme._automation._osam_session import OsamSession

        self._session = OsamSession(model_name=model_name)

    def shape_from_prompt(
        self,
        image: np.ndarray,
        image_id: str,
        prompt: PromptPoint,
        min_mask_area: int = 1,
    ) -> dict[str, Any] | None:
        import osam

        points = [(prompt.x, prompt.y), *prompt.negative_points]
        point_labels = [1] + [0] * len(prompt.negative_points)
        response: osam.types.GenerateResponse = self._session.run(
            image=image,
            image_id=image_id,
            points=np.array(points, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
        )

        annotations = sorted(
            response.annotations,
            key=lambda a: a.score if a.score is not None else 0.0,
            reverse=True,
        )
        for annotation in annotations:
            if annotation.mask is None or annotation.bounding_box is None:
                continue
            mask = annotation.mask.astype(bool)
            if int(mask.sum()) < min_mask_area:
                continue
            bb = annotation.bounding_box
            return {
                "label": prompt.label,
                "points": [
                    [float(bb.xmin), float(bb.ymin)],
                    [float(bb.xmax), float(bb.ymax)],
                ],
                "group_id": prompt.group_id,
                "description": prompt.description,
                "shape_type": "mask",
                "flags": {} if prompt.flags is None else dict(prompt.flags),
                "mask": encode_mask(mask),
            }
        return None


def coerce_prompt_point(value: PromptPoint | dict[str, Any]) -> PromptPoint:
    if isinstance(value, PromptPoint):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Prompt must be PromptPoint or dict, got {type(value)!r}")

    label = str(value["label"])
    if "point" in value:
        x, y = value["point"]
    else:
        x, y = value["x"], value["y"]

    negative_points_raw = value.get("negative_points") or []
    negative_points = tuple(
        (float(point[0]), float(point[1])) for point in negative_points_raw
    )
    flags = value.get("flags")
    if flags is not None and not isinstance(flags, dict):
        raise TypeError("flags must be a dict when provided")

    return PromptPoint(
        label=label,
        x=float(x),
        y=float(y),
        negative_points=negative_points,
        group_id=value.get("group_id"),
        description=str(value.get("description") or ""),
        flags=flags,
    )


def load_prompt_points(path: str | Path) -> list[PromptPoint]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "points" in data:
        data = data["points"]
    elif isinstance(data, dict):
        data = [
            {"label": label, "point": point}
            for label, points in data.items()
            for point in points
        ]
    if not isinstance(data, list):
        raise TypeError("Prompt JSON must be a list, {'points': list}, or label map")
    return [coerce_prompt_point(item) for item in data]


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"))


def make_labelme_image_path(image_path: Path, label_path: Path) -> str:
    try:
        return os.path.relpath(image_path, label_path.parent).replace(os.sep, "/")
    except ValueError:
        return image_path.as_posix()


def encode_mask(mask: np.ndarray) -> str:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_image_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def get_labelme_version() -> str:
    try:
        import labelme

        return str(labelme.__version__)
    except Exception:
        return "6.1.3"


def patch_osam_blob_path_for_windows() -> None:
    if os.name != "nt":
        return

    import osam.types._blob as blob_module

    current = blob_module.Blob.path.fget
    if getattr(current, "_labelme_yolo26s_windows_safe", False):
        return

    def windows_safe_path(self: Any) -> str:
        base = os.path.expanduser(
            os.path.join("~", ".cache", "osam", "models", "blobs")
        )
        safe_hash = self.hash.replace("sha256:", "sha256-")
        if self.attachments:
            return os.path.join(base, safe_hash, self.filename)
        return os.path.join(base, safe_hash)

    windows_safe_path._labelme_yolo26s_windows_safe = True  # type: ignore[attr-defined]
    blob_module.Blob.path = property(windows_safe_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LabelMe mask JSON from SAM2 point prompts."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--points",
        required=True,
        help=(
            "Prompt JSON. Accepts a list of {'label','point':[x,y]}, "
            "{'points': [...]}, or {'label': [[x,y], ...]}."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output LabelMe JSON path. Defaults to image path with .json suffix.",
    )
    parser.add_argument("--model", default="sam2:tiny", help="OSAM model name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output.")
    parser.add_argument(
        "--with-image-data",
        action="store_true",
        help="Embed imageData in the LabelMe JSON.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=1,
        help="Skip generated masks with fewer foreground pixels.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else image_path.with_suffix(".json")
    payload = generate_labelme_json_from_sam2_points(
        image_path=image_path,
        prompt_points=load_prompt_points(args.points),
        output_path=output_path,
        model_name=args.model,
        overwrite=args.overwrite,
        image_data=args.with_image_data,
        min_mask_area=args.min_mask_area,
    )
    print(
        f"Wrote {output_path} with {len(payload['shapes'])} shapes "
        f"for {payload['imageWidth']}x{payload['imageHeight']} image."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
