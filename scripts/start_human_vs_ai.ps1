param(
    [string]$Checkpoint = "D:\AI_data\cr-native-core\selfplay-v0.1\runs\selfplay-v0.1-stage-a-20260823T141402Z\evaluations\candidates\P010.pt",
    [int]$BattleSeed = 20260824,
    [int]$PolicySeed = 20260824,
    [string]$Python = "D:\AI_data\runtime\venv\Scripts\python.exe",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Training Python runtime not found: $Python"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "P010 checkpoint not found: $Checkpoint"
}

& (Join-Path $PSScriptRoot "build_probe.ps1") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Java host build failed: $LASTEXITCODE" }
& (Join-Path $PSScriptRoot "build_bridge.ps1") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Native bridge build failed: $LASTEXITCODE" }

& $Python -m native_core.worker start --workers 1 --base-port 37031 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Native Worker startup failed: $LASTEXITCODE" }

$Arguments = @(
    "-m", "native_core.human_vs_ai",
    "--checkpoint", $Checkpoint,
    "--battle-seed", "$BattleSeed",
    "--policy-seed", "$PolicySeed",
    "--port", "37031"
)
if ($Smoke) { $Arguments += "--smoke" }

if ($Smoke) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Human-vs-AI smoke failed: $LASTEXITCODE" }
    exit 0
}

$Pythonw = Join-Path (Split-Path -Parent $Python) "pythonw.exe"
if (-not (Test-Path -LiteralPath $Pythonw -PathType Leaf)) { $Pythonw = $Python }
Start-Process -FilePath $Pythonw -ArgumentList $Arguments -WorkingDirectory $ProjectRoot
Write-Host "Human-vs-P010 GUI started. You are Blue; P010 is Red."
