# 导出阶段 30：检测/分类模型导出为 OpenVINO
# 常用环境变量：TASK、WEIGHTS、CLASS_MODE、IMGSZ、BATCH、DYNAMIC、IMAGE_SIZE、DEVICE、OUTPUT_DIR、CLASSIFIER_MODEL、PRECISION
# 输出：runs/export_openvino 或 OUTPUT_DIR 下的 model.xml/model.bin
# 说明：类别、输入尺寸等写入 OpenVINO runtime metadata，JSON 仅作审计记录
# 当前生产路线：detector 默认 FP32；classifier 可先导出为中间模型，再用 32_quantize_classifier_int8.ps1 生成 INT8。

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Task = if ($env:TASK) { $env:TASK.ToLowerInvariant() } else { "detector" }
$Weights = if ($env:WEIGHTS) { $env:WEIGHTS } else { throw "请设置 WEIGHTS=权重路径" }
$ClassMode = if ($env:CLASS_MODE) { $env:CLASS_MODE } else { "single" }
$Imgsz = if ($env:IMGSZ) { $env:IMGSZ } else { "640" }
$Batch = if ($env:BATCH) { $env:BATCH } else { "1" }
$Dynamic = if ($env:DYNAMIC) { $env:DYNAMIC } else { "0" }
$ImageSize = if ($env:IMAGE_SIZE) { $env:IMAGE_SIZE } else { "0" }
$ClassifierModel = if ($env:CLASSIFIER_MODEL) { $env:CLASSIFIER_MODEL } else { "" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cpu" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "$Root\runs\export_openvino" }
$Precision = if ($env:PRECISION) { $env:PRECISION } else { if ($Task -eq "detector") { "fp32" } else { "fp16" } }
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"

Push-Location $Root
try {
  if ($Task -eq "detector") {
    $argsList = @(
      "-m", "pollen_pipeline.export_openvino",
      "--weights", $Weights,
      "--dataset-dir", "$Root\dataset",
      "--output-dir", $OutputDir,
      "--class-mode", $ClassMode,
      "--imgsz", $Imgsz,
      "--batch", $Batch,
      "--device", $Device,
      "--precision", $Precision
    )
    if ($Dynamic -match "^(1|true|yes)$") {
      $argsList += "--dynamic"
    }
    & $Python @argsList
  }
  elseif ($Task -eq "classifier") {
    $argsList = @(
      "-m", "pollen_pipeline.export_classifier_openvino",
      "--weights", $Weights,
      "--output-dir", $OutputDir,
      "--image-size", $ImageSize,
      "--device", $Device
    )
    if ($ClassifierModel) {
      $argsList += @("--model", $ClassifierModel)
    }
    & $Python @argsList
  }
  else {
    throw "TASK 只能是 detector 或 classifier，当前：$Task"
  }
}
finally {
  Pop-Location
}
