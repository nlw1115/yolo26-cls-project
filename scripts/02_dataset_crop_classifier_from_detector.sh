#!/usr/bin/env bash
# 数据集阶段 02：使用检测权重全量推理，并离线裁剪分类训练图片
# 输入：dataset/fused_labelme、检测权重 WEIGHTS
# 输出：derived/detector_predictions/<RUN_ID>/manifest.csv
# 输出：derived/classification_crops/<RUN_ID>/train|val/class_*/...
# 常用环境变量：WEIGHTS、CONF、DEVICE、RUN_ID、CROP_SCALE、IMAGE_SIZE、IMGSZ、IOU、MATCH_IOU、MAX_DET

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

if [[ -z "${WEIGHTS:-}" ]]; then
  echo "请设置 WEIGHTS=检测权重路径，例如 runs/detector/xxx/weights/best.pt" >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.crop_classifier_from_detector \
  --weights "$WEIGHTS" \
  --dataset-dir "$ROOT/dataset" \
  --run-id "${RUN_ID:-}" \
  --predictions-dir "$ROOT/derived/detector_predictions" \
  --output-root "$ROOT/derived/classification_crops" \
  --device "${DEVICE:-cpu}" \
  --conf "${CONF:-0.35}" \
  --iou "${IOU:-0.6}" \
  --match-iou "${MATCH_IOU:-0.5}" \
  --max-det "${MAX_DET:-300}" \
  --imgsz "${IMGSZ:-640}" \
  --crop-scale "${CROP_SCALE:-1.6}" \
  --image-size "${IMAGE_SIZE:-224}" \
  --jpeg-quality "${JPEG_QUALITY:-95}"
