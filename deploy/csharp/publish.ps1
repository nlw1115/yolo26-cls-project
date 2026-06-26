param(
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-WithinDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $resolvedRoot = [System.IO.Path]::GetFullPath($RootPath).TrimEnd('\')
    $resolvedTarget = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $resolvedTarget.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify path outside root. Root=$resolvedRoot Target=$resolvedTarget"
    }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$distRoot = Join-Path $scriptRoot "dist"
$algorithmDist = Join-Path $distRoot "algorithm"
$demoDist = Join-Path $distRoot "demo"
$algorithmProject = Join-Path $scriptRoot "PollenInference\\PollenInference.csproj"
$demoProject = Join-Path $scriptRoot "PollenInferenceDemo\\PollenInferenceDemo.csproj"

Assert-WithinDirectory -RootPath $scriptRoot -TargetPath $distRoot
Assert-WithinDirectory -RootPath $scriptRoot -TargetPath $algorithmDist
Assert-WithinDirectory -RootPath $scriptRoot -TargetPath $demoDist

if (Test-Path -LiteralPath $algorithmDist) {
    Remove-Item -LiteralPath $algorithmDist -Recurse -Force
}

if (Test-Path -LiteralPath $demoDist) {
    Remove-Item -LiteralPath $demoDist -Recurse -Force
}

New-Item -ItemType Directory -Path $algorithmDist | Out-Null
New-Item -ItemType Directory -Path $demoDist | Out-Null

dotnet build $algorithmProject -c $Configuration

$algorithmDll = Join-Path $scriptRoot "PollenInference\\bin\\$Configuration\\net10.0-windows\\PollenInference.dll"
if (-not (Test-Path -LiteralPath $algorithmDll)) {
    throw "Algorithm dll was not produced: $algorithmDll"
}

Copy-Item -LiteralPath $algorithmDll -Destination (Join-Path $algorithmDist "PollenInference.dll") -Force

dotnet publish $demoProject `
    -c $Configuration `
    -r $RuntimeIdentifier `
    --self-contained true `
    -p:PublishSingleFile=true `
    -p:EnableCompressionInSingleFile=true `
    -p:IncludeNativeLibrariesForSelfExtract=true `
    -p:DebugType=None `
    -p:DebugSymbols=false `
    -p:PublishTrimmed=false `
    -o $demoDist

Get-ChildItem -LiteralPath $demoDist -File |
    Where-Object { $_.Name -ne "PollenInferenceDemo.exe" } |
    Remove-Item -Force

Get-ChildItem -LiteralPath $demoDist -Directory |
    Remove-Item -Recurse -Force

$version = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($algorithmDll).ProductVersion
Write-Host "Algorithm dll: $algorithmDist\\PollenInference.dll"
Write-Host "Algorithm version: $version"
Write-Host "Demo exe directory: $demoDist"
