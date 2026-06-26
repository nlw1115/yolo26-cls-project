#!/usr/bin/env bash
# 评估阶段 22：检测 + 分类级联评估
# 输入：DETECTOR_WEIGHTS、CLASSIFIER_WEIGHTS
# 输出：runs/pipeline_eval/<RUN_NAME>/pipeline_predictions.csv、pipeline_report.json、混淆矩阵

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
: "${DETECTOR_WEIGHTS:?请设置 DETECTOR_WEIGHTS=检测权重路径}"
: "${CLASSIFIER_WEIGHTS:?请设置 CLASSIFIER_WEIGHTS=分类权重路径}"

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.pipeline_eval \
  --detector-weights "$DETECTOR_WEIGHTS" \
  --classifier-weights "$CLASSIFIER_WEIGHTS" \
  --dataset-dir "$ROOT/dataset" \
  --output-dir "$ROOT/runs/pipeline_eval/${RUN_NAME:-pipeline_eval}" \
  --device "${DEVICE:-cpu}" \
  --conf "${CONF:-0.35}" \
  --crop-scale "${CROP_SCALE:-1.6}"

