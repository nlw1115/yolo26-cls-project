#!/usr/bin/env bash
# 训练阶段 10：单类别生产检测训练
# 输入：dataset/fused_labelme/*.json，训练时内存合并为 pollen，不生成 YOLO 标签文件
# 输出：runs/detector/<RUN_NAME>/weights/best.pt
# 常用环境变量：EPOCHS、BATCH、DEVICE、WORKERS、IMGsz、RUN_NAME、AUGMENT

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

RUN_NAME="${RUN_NAME:-singlecls_aug_mild_box10_seed_26052016}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
IMGSZ="${IMGSZ:-640}"
DATA_REPEAT="${DATA_REPEAT:-4}"

cd "$ROOT"
"$PYTHON" -m pollen_pipeline.detector_train \
  --dataset-dir "$ROOT/dataset" \
  --class-mode single \
  --single-class-name pollen \
  --model yolo26s.yaml \
  --pretrained auto \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --data-repeat "$DATA_REPEAT" \
  --lr0 "${LR0:-0.001}" \
  --lrf "${LRF:-0.02}" \
  --box "${BOX:-10.0}" \
  --cls "${CLS:-0.25}" \
  --dfl "${DFL:-2.0}" \
  --weight-decay "${WEIGHT_DECAY:-0.001}" \
  --warmup-bias-lr "${WARMUP_BIAS_LR:-0.01}" \
  --seed "${SEED:-26052016}" \
  --augment "${AUGMENT:-true}" \
  --degrees "${DEGREES:-5}" \
  --translate "${TRANSLATE:-0.03}" \
  --scale "${SCALE:-0.10}" \
  --flipud "${FLIPUD:-0.5}" \
  --fliplr "${FLIPLR:-0.5}" \
  --hsv-s "${HSV_S:-0.03}" \
  --hsv-v "${HSV_V:-0.05}" \
  --exist-ok

