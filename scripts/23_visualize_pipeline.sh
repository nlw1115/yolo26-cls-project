#!/usr/bin/env bash
# 评估阶段 23：检测 + 分类级联可视化
# 输入：DETECTOR_WEIGHTS、CLASSIFIER_WEIGHTS
# 输出：runs/pipeline_visualize/<RUN_NAME>/*.jpg

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
: "${DETECTOR_WEIGHTS:?请设置 DETECTOR_WEIGHTS=检测权重路径}"
: "${CLASSIFIER_WEIGHTS:?请设置 CLASSIFIER_WEIGHTS=分类权重路径}"

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.visualize_pipeline \
  --detector-weights "$DETECTOR_WEIGHTS" \
  --classifier-weights "$CLASSIFIER_WEIGHTS" \
  --dataset-dir "$ROOT/dataset" \
  --output-dir "$ROOT/runs/pipeline_visualize/${RUN_NAME:-pipeline_visualize}" \
  --device "${DEVICE:-cpu}" \
  --conf "${CONF:-0.35}" \
  --limit "${LIMIT:-0}"

