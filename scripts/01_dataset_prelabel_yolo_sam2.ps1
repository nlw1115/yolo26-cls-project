# 数据集阶段 01：YOLO 粗检中心点 + SAM2 生成待复核 LabelMe 预标注
# 常用环境变量：WEIGHTS、CONF、DEVICE、SAM2_MODEL、SUBSET、RUN_ID
# 输入：dataset/fused_labelme、检测权重
# 输出：derived/prelabels/<RUN_ID>/*.json
# 注意：不自动覆盖 dataset/fused_labelme，人工复核后再合并

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Weights = if ($env:WEIGHTS) { $env:WEIGHTS } else { throw "请设置 WEIGHTS=多类别检测权重路径" }
$Conf = if ($env:CONF) { $env:CONF } else { "0.25" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cpu" }
$Sam2Model = if ($env:SAM2_MODEL) { $env:SAM2_MODEL } else { "sam2:tiny" }
$Subset = if ($env:SUBSET) { $env:SUBSET } else { "all" }
$RunId = if ($env:RUN_ID) { $env:RUN_ID } else { "" }
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"

Push-Location $Root
try {
  & $Python -m pollen_pipeline.prelabel_yolo_sam2 `
    --weights $Weights `
    --dataset-dir "$Root\dataset" `
    --subset $Subset `
    --output-root "$Root\derived\prelabels" `
    --run-id $RunId `
    --sam2-model $Sam2Model `
    --device $Device `
    --conf $Conf
}
finally {
  Pop-Location
}

