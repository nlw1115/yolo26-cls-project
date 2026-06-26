#!/usr/bin/env bash
# 训练阶段 12：训练离线 crop 二级分类模型
# 输入：derived/classification_crops/<CROP_RUN>/train|val/class_*
# 输出：runs/classifier/<RUN_NAME>/weights/best.pt
# 常用环境变量：CROP_RUN 或 CLASSIFIER_DATASET、EPOCHS、BATCH、MODEL、DEVICE

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

if [[ -n "${CLASSIFIER_DATASET:-}" ]]; then
  DATASET="$CLASSIFIER_DATASET"
elif [[ -n "${CROP_RUN:-}" ]]; then
  DATASET="$ROOT/derived/classification_crops/$CROP_RUN"
else
  echo "请设置 CROP_RUN 或 CLASSIFIER_DATASET" >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.classifier_train \
  --dataset-dir "$DATASET" \
  --runs-dir "$ROOT/runs/classifier" \
  --run-name "${RUN_NAME:-classify_fused_resnet18}" \
  --model "${MODEL:-resnet18}" \
  --epochs "${EPOCHS:-100}" \
  --batch "${BATCH:-32}" \
  --workers "${WORKERS:-4}" \
  --device "${DEVICE:-0}" \
  --lr "${LR:-0.0003}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --pretrained "${PRETRAINED:-true}" \
  --augment "${AUGMENT:-true}" \
  --exist-ok

