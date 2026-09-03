param(
    [string]$Checkpoint = "D:\AI_data\cr-native-core\selfplay-v0.2\runs\selfplay-v0.2-scratch-5m-20260824T023123Z\evaluations\candidates\P050.pt",
    [int]$BattleSeed = 20260824,
    [int]$PolicySeed = 20260824,
    [string]$Python = "D:\AI_data\runtime\venv\Scripts\python.exe",
    [string]$ExpertDatasetRoot = "D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3-current-frontier-v5\compiled\native-bc-v1",
    [string]$Replay = "examples\eight-card-bootstrap.json",
    [double]$ExpertPlayRateScale = 1.0,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Training Python runtime not found: $Python"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "AI checkpoint not found: $Checkpoint"
}
if (-not (Test-Path -LiteralPath $Replay -PathType Leaf)) {
    throw "Battle deck preset not found: $Replay"
}
$Replay = (Resolve-Path -LiteralPath $Replay).Path

Write-Host "[1/4] Building Java host..."
& (Join-Path $PSScriptRoot "build_probe.ps1") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Java host build failed: $LASTEXITCODE" }
Write-Host "[2/4] Building native bridge..."
& (Join-Path $PSScriptRoot "build_bridge.ps1") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Native bridge build failed: $LASTEXITCODE" }

Write-Host "[3/4] Starting Android worker and native battle service..."
& $Python -u -m native_core.worker start --workers 1 --base-port 37031 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Native Worker startup failed: $LASTEXITCODE" }

$Arguments = @(
    "-m", "native_core.human_vs_ai",
    "--checkpoint", $Checkpoint,
    "--expert-dataset-root", $ExpertDatasetRoot,
    "--replay", $Replay,
    "--battle-seed", "$BattleSeed",
    "--policy-seed", "$PolicySeed",
    "--expert-play-rate-scale", $ExpertPlayRateScale.ToString([Globalization.CultureInfo]::InvariantCulture),
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
$LogDirectory = "D:\AI_data\cr-native-core\human-vs-ai\logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogStamp = (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N").Substring(0, 6)
Write-Host "[4/4] Loading the selected checkpoint and opening the match..."
Start-Process -FilePath $Pythonw -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -RedirectStandardOutput (Join-Path $LogDirectory "$LogStamp.stdout.log") -RedirectStandardError (Join-Path $LogDirectory "$LogStamp.stderr.log")
Write-Host "Human-vs-AI GUI started. You are Blue; the loaded checkpoint is Red."
Write-Host "Startup logs: $LogDirectory"
