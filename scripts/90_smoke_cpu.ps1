# 验证入口 90：本地 CPU 快速检查 v3 项目是否能导入、读取 LabelMe 数据和显示 CLI help
# 不训练，不写 runs；用于确认环境和脚本路径

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONPATH = "$Root\src;$env:PYTHONPATH"

Push-Location $Root
try {
  & $Python -m pollen_pipeline.detector_train --help | Out-Null
  & $Python -m pollen_pipeline.detector_eval --help | Out-Null
  & $Python -m pollen_pipeline.crop_classifier_from_detector --help | Out-Null
  & $Python -c "from pathlib import Path; from pollen_pipeline.labelme_dataset import load_bundle; b=load_bundle(Path('dataset'), class_mode='single'); print('train', len(b.train_records), 'val', len(b.val_records), 'classes', b.class_names)"
}
finally {
  Pop-Location
}

