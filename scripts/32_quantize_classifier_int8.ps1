# 导出阶段 32：将分类器 OpenVINO 模型做 INT8 PTQ
# 常用环境变量：MODEL_XML、OUTPUT_DIR、IMAGE_SIZE、CROP_SCALE、SAMPLES_PER_CLASS、SUBSET_SIZE、PYTHON
# 输入：已导出的分类器 OpenVINO model.xml 与 dataset/fused_labelme LabelMe 标注
# 输出：OUTPUT_DIR/model_int8.xml、model_int8.bin、quantize_meta.json

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$ModelXml = if ($env:MODEL_XML) { $env:MODEL_XML } else { throw "请设置 MODEL_XML=分类器 OpenVINO model.xml" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "$Root\runs\export_openvino_classifier_int8" }
$ImageSize = if ($env:IMAGE_SIZE) { $env:IMAGE_SIZE } else { "224" }
$CropScale = if ($env:CROP_SCALE) { $env:CROP_SCALE } else { "1.7" }
$SamplesPerClass = if ($env:SAMPLES_PER_CLASS) { $env:SAMPLES_PER_CLASS } else { "120" }
$SubsetSize = if ($env:SUBSET_SIZE) { $env:SUBSET_SIZE } else { "300" }
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"

Push-Location $Root
try {
  & $Python -m pollen_pipeline.quantize_classifier_openvino `
    --model-xml $ModelXml `
    --dataset-dir "$Root\dataset" `
    --output-dir $OutputDir `
    --image-size $ImageSize `
    --crop-scale $CropScale `
    --samples-per-class $SamplesPerClass `
    --subset-size $SubsetSize
}
finally {
  Pop-Location
}
