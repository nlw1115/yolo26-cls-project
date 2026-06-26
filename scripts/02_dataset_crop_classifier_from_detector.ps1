# 数据集阶段 02：使用检测权重全量推理并离线裁剪分类训练图片
# 常用环境变量：WEIGHTS、CONF、DEVICE、RUN_ID、CROP_SCALE、IMAGE_SIZE、IMGSZ、IOU、MATCH_IOU、MAX_DET
# 输入：dataset/fused_labelme、检测权重
# 输出：derived/detector_predictions/<RUN_ID>/manifest.csv
# 输出：derived/classification_crops/<RUN_ID>/train|val/class_*/...

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Weights = if ($env:WEIGHTS) { $env:WEIGHTS } else { throw "请设置 WEIGHTS=检测权重路径" }
$Conf = if ($env:CONF) { $env:CONF } else { "0.35" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cpu" }
$RunId = if ($env:RUN_ID) { $env:RUN_ID } else { "" }
$CropScale = if ($env:CROP_SCALE) { $env:CROP_SCALE } else { "1.6" }
$ImageSize = if ($env:IMAGE_SIZE) { $env:IMAGE_SIZE } else { "224" }
$ImgSz = if ($env:IMGSZ) { $env:IMGSZ } else { "640" }
$Iou = if ($env:IOU) { $env:IOU } else { "0.6" }
$MatchIou = if ($env:MATCH_IOU) { $env:MATCH_IOU } else { "0.5" }
$MaxDet = if ($env:MAX_DET) { $env:MAX_DET } else { "300" }
$JpegQuality = if ($env:JPEG_QUALITY) { $env:JPEG_QUALITY } else { "95" }
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"

Push-Location $Root
try {
  & $Python -m pollen_pipeline.crop_classifier_from_detector `
    --weights $Weights `
    --dataset-dir "$Root\dataset" `
    --run-id $RunId `
    --predictions-dir "$Root\derived\detector_predictions" `
    --output-root "$Root\derived\classification_crops" `
    --device $Device `
    --conf $Conf `
    --iou $Iou `
    --match-iou $MatchIou `
    --max-det $MaxDet `
    --imgsz $ImgSz `
    --crop-scale $CropScale `
    --image-size $ImageSize `
    --jpeg-quality $JpegQuality
}
finally {
  Pop-Location
}
