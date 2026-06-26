#!/usr/bin/env bash
# 评估阶段 20：检测模型评估与 overlay 复核图
# 颜色：绿色=GT，蓝色=TP，红色=FP；FN 表现为只有绿框没有蓝框
# 输入：WEIGHTS 检测权重
# 输出：runs/detector_eval/<RUN_NAME>/metrics.csv、overlays/*.jpg、detection_review.md

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
: "${WEIGHTS:?请设置 WEIGHTS=检测权重路径}"

OUT="$ROOT/runs/detector_eval/${RUN_NAME:-$(basename "$(dirname "$(dirname "$WEIGHTS")")")_eval}"
ARGS=(
  --weights "$WEIGHTS"
  --dataset-dir "$ROOT/dataset"
  --subset "${SUBSET:-val}"
  --class-mode "${CLASS_MODE:-single}"
  --output-dir "$OUT"
  --device "${DEVICE:-cpu}"
  --imgsz "${IMGSZ:-640}"
  --conf "${CONF:-0.35}"
  --ap-conf "${AP_CONF:-0.001}"
  --nms-iou "${NMS_IOU:-0.6}"
  --match-iou "${MATCH_IOU:-0.5}"
)

if [[ "${CLASS_AWARE:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]] || [[ "${CLASS_MODE:-single}" == "multi" ]]; then
  ARGS+=(--class-aware)
fi

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.detector_eval "${ARGS[@]}"
