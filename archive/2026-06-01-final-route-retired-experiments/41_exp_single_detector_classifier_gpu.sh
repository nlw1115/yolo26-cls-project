#!/usr/bin/env bash
# 对比实验 B：单类别目标检出 + 多类别分类模型级联判别。
# 输入：dataset/fused_labelme/*.json、dataset/classes.yaml、dataset/split.json。
# 输出：单类检测权重、离线 crop 数据、分类权重，以及 runs/compare_eval/<RUN_NAME>/metrics.csv。
# 设计要点：检测阶段弱化类别损失、强化 box/dfl 和召回；分类阶段用 WeightedRandomSampler 与类别权重处理不均衡，并对 crop 做温和小目标增强。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

RUN_NAME="${RUN_NAME:-expB_single_det_cls_seed26052016}"
DET_RUN_NAME="${DET_RUN_NAME:-${RUN_NAME}_detector}"
CROP_RUN="${CROP_RUN:-${RUN_NAME}_crops_conf035}"
CLS_RUN_NAME="${CLS_RUN_NAME:-${RUN_NAME}_classifier}"
MODEL="${MODEL:-yolo26s.yaml}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-resnet34}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-26052016}"

cd "$ROOT"

"$PYTHON" -m pollen_pipeline.detector_train \
  --dataset-dir "$ROOT/dataset" \
  --class-mode single \
  --single-class-name pollen \
  --model "$MODEL" \
  --pretrained "${PRETRAINED:-auto}" \
  --runs-dir "$ROOT/runs/detector" \
  --run-name "$DET_RUN_NAME" \
  --epochs "${DET_EPOCHS:-120}" \
  --batch "${DET_BATCH:-12}" \
  --imgsz "${DET_IMGSZ:-960}" \
  --device "$DEVICE" \
  --workers "${WORKERS:-8}" \
  --data-repeat "${DET_DATA_REPEAT:-5}" \
  --optimizer "${DET_OPTIMIZER:-AdamW}" \
  --lr0 "${DET_LR0:-0.001}" \
  --lrf "${DET_LRF:-0.02}" \
  --weight-decay "${DET_WEIGHT_DECAY:-0.001}" \
  --warmup-epochs "${DET_WARMUP_EPOCHS:-4}" \
  --warmup-bias-lr "${DET_WARMUP_BIAS_LR:-0.01}" \
  --box "${DET_BOX:-12.0}" \
  --cls "${DET_CLS:-0.20}" \
  --dfl "${DET_DFL:-2.5}" \
  --patience "${DET_PATIENCE:-20}" \
  --amp "${AMP:-true}" \
  --cache "${CACHE:-false}" \
  --seed "$SEED" \
  --augment "${DET_AUGMENT:-true}" \
  --mosaic "${DET_MOSAIC:-0.10}" \
  --mixup "${DET_MIXUP:-0.0}" \
  --degrees "${DET_DEGREES:-5}" \
  --translate "${DET_TRANSLATE:-0.03}" \
  --scale "${DET_SCALE:-0.12}" \
  --shear "${DET_SHEAR:-0.0}" \
  --perspective "${DET_PERSPECTIVE:-0.0}" \
  --flipud "${DET_FLIPUD:-0.5}" \
  --fliplr "${DET_FLIPLR:-0.5}" \
  --hsv-h "${DET_HSV_H:-0.0}" \
  --hsv-s "${DET_HSV_S:-0.03}" \
  --hsv-v "${DET_HSV_V:-0.05}" \
  --plots "${PLOTS:-false}" \
  --show-class-metrics false \
  --exist-ok

DET_WEIGHTS="$ROOT/runs/detector/$DET_RUN_NAME/weights/best.pt"

"$PYTHON" -m pollen_pipeline.crop_classifier_from_detector \
  --weights "$DET_WEIGHTS" \
  --dataset-dir "$ROOT/dataset" \
  --run-id "$CROP_RUN" \
  --predictions-dir "$ROOT/derived/detector_predictions" \
  --output-root "$ROOT/derived/classification_crops" \
  --device "$DEVICE" \
  --conf "${CROP_CONF:-0.35}" \
  --iou "${CROP_NMS_IOU:-0.6}" \
  --match-iou "${CROP_MATCH_IOU:-0.5}" \
  --max-det "${MAX_DET:-300}" \
  --imgsz "${CROP_IMGSZ:-${DET_IMGSZ:-960}}" \
  --crop-scale "${CROP_SCALE:-1.7}" \
  --image-size "${IMAGE_SIZE:-224}" \
  --jpeg-quality "${JPEG_QUALITY:-95}"

"$PYTHON" -m pollen_pipeline.classifier_train \
  --dataset-dir "$ROOT/derived/classification_crops/$CROP_RUN" \
  --runs-dir "$ROOT/runs/classifier" \
  --run-name "$CLS_RUN_NAME" \
  --model "$CLASSIFIER_MODEL" \
  --epochs "${CLS_EPOCHS:-100}" \
  --batch "${CLS_BATCH:-64}" \
  --workers "${WORKERS:-8}" \
  --device "$DEVICE" \
  --image-size "${IMAGE_SIZE:-224}" \
  --crop-scale "${CROP_SCALE:-1.7}" \
  --lr "${CLS_LR:-0.0003}" \
  --weight-decay "${CLS_WEIGHT_DECAY:-0.0002}" \
  --seed "$SEED" \
  --pretrained "${CLS_PRETRAINED:-true}" \
  --augment "${CLS_AUGMENT:-true}" \
  --exist-ok

CLS_WEIGHTS="$ROOT/runs/classifier/$CLS_RUN_NAME/weights/best.pt"
OUT_DIR="$ROOT/runs/compare_eval/$RUN_NAME"

"$PYTHON" -m pollen_pipeline.pipeline_eval \
  --detector-weights "$DET_WEIGHTS" \
  --classifier-weights "$CLS_WEIGHTS" \
  --dataset-dir "$ROOT/dataset" \
  --subset val \
  --output-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --imgsz "${EVAL_IMGSZ:-${DET_IMGSZ:-960}}" \
  --conf "${EVAL_CONF:-0.35}" \
  --ap-conf "${AP_CONF:-0.001}" \
  --nms-iou "${NMS_IOU:-0.6}" \
  --match-iou "${MATCH_IOU:-0.5}" \
  --max-det "${MAX_DET:-300}" \
  --crop-scale "${CROP_SCALE:-1.7}" \
  --image-size "${IMAGE_SIZE:-224}"

echo "单类检出 + 分类级联实验完成："
echo "  detector:   $DET_WEIGHTS"
echo "  classifier: $CLS_WEIGHTS"
echo "验证集四项指标见：$OUT_DIR/metrics.csv"
