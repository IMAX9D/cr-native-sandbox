param(
    [int]$Port = 37031,
    [switch]$Smoke,
    [string]$Python = "D:\AI_data\runtime\venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Training Python runtime not found: $Python"
}

& (Join-Path $PSScriptRoot "build_probe.ps1") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Java host build failed: $LASTEXITCODE" }
& (Join-Path $PSScriptRoot "build_bridge.ps1") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Native bridge build failed: $LASTEXITCODE" }

& $Python -m native_core.worker start --workers 1 --base-port $Port | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Native Worker startup failed: $LASTEXITCODE" }

$Arguments = @("-m", "native_core.gui", "--port", "$Port")
if ($Smoke) {
    $Arguments += "--smoke"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Logic GUI smoke failed: $LASTEXITCODE" }
    exit 0
}

$Pythonw = Join-Path (Split-Path -Parent $Python) "pythonw.exe"
if (-not (Test-Path -LiteralPath $Pythonw -PathType Leaf)) { $Pythonw = $Python }
Start-Process -FilePath $Pythonw -ArgumentList $Arguments -WorkingDirectory $ProjectRoot
Write-Host "Game logic acceptance GUI started."
