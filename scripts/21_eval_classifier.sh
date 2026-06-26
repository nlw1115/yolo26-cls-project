#!/usr/bin/env bash
# 评估阶段 21：评估二级分类模型
# 输入：WEIGHTS 分类权重，CROP_RUN 或 CLASSIFIER_DATASET
# 输出：runs/classifier_eval/<RUN_NAME>/confusion_matrix.* 与 classification_eval.json

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
: "${WEIGHTS:?请设置 WEIGHTS=分类权重路径}"
if [[ -n "${CLASSIFIER_DATASET:-}" ]]; then
  DATASET="$CLASSIFIER_DATASET"
elif [[ -n "${CROP_RUN:-}" ]]; then
  DATASET="$ROOT/derived/classification_crops/$CROP_RUN"
else
  echo "请设置 CROP_RUN 或 CLASSIFIER_DATASET" >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.classifier_eval \
  --weights "$WEIGHTS" \
  --dataset-dir "$DATASET" \
  --output-dir "$ROOT/runs/classifier_eval/${RUN_NAME:-classifier_eval}" \
  --device "${DEVICE:-cpu}" \
  --batch "${BATCH:-32}"

