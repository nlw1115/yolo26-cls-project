# 导出阶段 31：打包 C# 算法核心与 demo
# 输入：deploy/csharp 下的 C# 工程
# 输出：deploy/csharp/dist
# 当前 v3 先保留标准入口；若 C# 工程尚未迁入，会明确报错而不静默跳过

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Publish = "$Root\deploy\csharp\publish.ps1"
if (-not (Test-Path -LiteralPath $Publish)) {
  throw "尚未迁入 C# 部署工程：$Publish"
}
Push-Location $Root
try {
  & pwsh $Publish
}
finally {
  Pop-Location
}

