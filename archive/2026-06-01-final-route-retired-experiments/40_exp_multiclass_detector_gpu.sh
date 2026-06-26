#!/usr/bin/env bash
# 对比实验 A：多类别目标检测直接完成花粉多分类。
# 输入：dataset/fused_labelme/*.json、dataset/classes.yaml、dataset/split.json。
# 输出：runs/detector/<RUN_NAME>/weights/best.pt 与 runs/compare_eval/<RUN_NAME>/metrics.csv。
# 设计要点：保留多类别；用类均衡重复采样和类别损失权重缓解样本不均衡；用高分辨率和温和几何/颜色增强照顾花粉小目标。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

RUN_NAME="${RUN_NAME:-expA_multiclass_det_yolo26s_seed26052016}"
MODEL="${MODEL:-yolo26s.yaml}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-26052016}"

cd "$ROOT"

"$PYTHON" -m pollen_pipeline.detector_train \
  --dataset-dir "$ROOT/dataset" \
  --class-mode multi \
  --model "$MODEL" \
  --pretrained "${PRETRAINED:-auto}" \
  --runs-dir "$ROOT/runs/detector" \
  --run-name "$RUN_NAME" \
  --epochs "${EPOCHS:-120}" \
  --batch "${BATCH:-8}" \
  --imgsz "${IMGSZ:-960}" \
  --device "$DEVICE" \
  --workers "${WORKERS:-8}" \
  --data-repeat "${DATA_REPEAT:-6}" \
  --class-balance-alpha "${CLASS_BALANCE_ALPHA:-0.5}" \
  --class-balance-max-repeat "${CLASS_BALANCE_MAX_REPEAT:-5}" \
  --class-weight-alpha "${CLASS_WEIGHT_ALPHA:-0.5}" \
  --class-weight-max "${CLASS_WEIGHT_MAX:-3.0}" \
  --optimizer "${OPTIMIZER:-AdamW}" \
  --lr0 "${LR0:-0.0008}" \
  --lrf "${LRF:-0.02}" \
  --weight-decay "${WEIGHT_DECAY:-0.001}" \
  --warmup-epochs "${WARMUP_EPOCHS:-5}" \
  --warmup-bias-lr "${WARMUP_BIAS_LR:-0.01}" \
  --box "${BOX:-10.0}" \
  --cls "${CLS:-0.65}" \
  --dfl "${DFL:-2.0}" \
  --patience "${PATIENCE:-20}" \
  --amp "${AMP:-true}" \
  --cache "${CACHE:-false}" \
  --seed "$SEED" \
  --augment "${AUGMENT:-true}" \
  --mosaic "${MOSAIC:-0.15}" \
  --mixup "${MIXUP:-0.0}" \
  --degrees "${DEGREES:-7}" \
  --translate "${TRANSLATE:-0.04}" \
  --scale "${SCALE:-0.18}" \
  --shear "${SHEAR:-0.0}" \
  --perspective "${PERSPECTIVE:-0.0}" \
  --flipud "${FLIPUD:-0.5}" \
  --fliplr "${FLIPLR:-0.5}" \
  --hsv-h "${HSV_H:-0.0}" \
  --hsv-s "${HSV_S:-0.04}" \
  --hsv-v "${HSV_V:-0.06}" \
  --plots "${PLOTS:-false}" \
  --show-class-metrics "${SHOW_CLASS_METRICS:-true}" \
  --exist-ok

WEIGHTS="$ROOT/runs/detector/$RUN_NAME/weights/best.pt"
OUT_DIR="$ROOT/runs/compare_eval/$RUN_NAME"

"$PYTHON" -m pollen_pipeline.detector_eval \
  --weights "$WEIGHTS" \
  --dataset-dir "$ROOT/dataset" \
  --subset val \
  --class-mode multi \
  --output-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --imgsz "${EVAL_IMGSZ:-${IMGSZ:-960}}" \
  --conf "${EVAL_CONF:-0.35}" \
  --ap-conf "${AP_CONF:-0.001}" \
  --nms-iou "${NMS_IOU:-0.6}" \
  --match-iou "${MATCH_IOU:-0.5}" \
  --max-det "${MAX_DET:-300}" \
  --class-aware

echo "多类别检测实验完成：$WEIGHTS"
echo "验证集四项指标见：$OUT_DIR/metrics.csv"
