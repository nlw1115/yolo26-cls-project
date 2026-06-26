#!/usr/bin/env bash
# 训练阶段 11：多类别预标注粗模型训练
# 输入：dataset/fused_labelme/*.json，保留原始多类别
# 输出：runs/detector/<RUN_NAME>/weights/best.pt
# 用途：小数据快速收敛，供 01_dataset_prelabel_yolo_sam2.ps1 产生 SAM2 提示点

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.detector_train \
  --dataset-dir "$ROOT/dataset" \
  --class-mode multi \
  --model yolo26s.yaml \
  --pretrained auto \
  --run-name "${RUN_NAME:-multi_prelabel_fast}" \
  --epochs "${EPOCHS:-20}" \
  --batch "${BATCH:-8}" \
  --imgsz "${IMGSZ:-640}" \
  --device "${DEVICE:-0}" \
  --workers "${WORKERS:-4}" \
  --data-repeat "${DATA_REPEAT:-8}" \
  --class-balance-alpha "${CLASS_BALANCE_ALPHA:-0}" \
  --class-balance-max-repeat "${CLASS_BALANCE_MAX_REPEAT:-4}" \
  --class-weight-alpha "${CLASS_WEIGHT_ALPHA:-0}" \
  --class-weight-max "${CLASS_WEIGHT_MAX:-3}" \
  --lr0 "${LR0:-0.003}" \
  --lrf "${LRF:-0.05}" \
  --box "${BOX:-10.0}" \
  --cls "${CLS:-0.25}" \
  --dfl "${DFL:-2.0}" \
  --weight-decay "${WEIGHT_DECAY:-0.001}" \
  --augment "${AUGMENT:-true}" \
  --plots "${PLOTS:-false}" \
  --show-class-metrics "${SHOW_CLASS_METRICS:-true}" \
  --exist-ok
