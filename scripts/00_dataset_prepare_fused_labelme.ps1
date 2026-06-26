# 数据集阶段 00：整理 fused_labelme 训练目录
# 输入：dataset/groups/<视场>/fused.jpg 与 labelme.json
# 输出：dataset/fused_labelme/<视场>.jpg 与 <视场>.json
# 写入：dataset/fused_labelme；不写 YOLO txt，不写 annotations.jsonl

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"

Push-Location $Root
try {
  & $Python -m pollen_pipeline.prepare_fused_labelme_dataset `
    --dataset-dir "$Root\dataset"
}
finally {
  Pop-Location
}

