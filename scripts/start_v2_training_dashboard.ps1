param(
    [int]$Port = 8765,
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

$LogRoot = "D:\AI_data\cr-native-core\selfplay-v0.2\dashboard\launcher-logs"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Stdout = Join-Path $LogRoot "$Stamp-v2-dashboard-stdout.log"
$Stderr = Join-Path $LogRoot "$Stamp-v2-dashboard-stderr.log"
$Arguments = @(
    (Join-Path $PSScriptRoot "run_v2_training_dashboard.py"),
    "--port", "$Port"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr

$Deadline = (Get-Date).AddSeconds(20)
$Url = "http://127.0.0.1:$Port/"
do {
    Start-Sleep -Milliseconds 250
    if ($Process.HasExited) {
        $Text = if (Test-Path $Stderr) { Get-Content -Raw $Stderr } else { "" }
        throw "v0.2 dashboard exited during startup: $Text"
    }
    try {
        $State = Invoke-RestMethod -Uri "$Url/api/state" -TimeoutSec 1
    } catch {
        $State = $null
    }
} while ($null -eq $State -and (Get-Date) -lt $Deadline)
if ($null -eq $State) { throw "v0.2 dashboard did not answer within 20 seconds" }
[pscustomobject]@{
    pid = $Process.Id
    url = $Url
    phase = $State.phase_label
    run_id = $State.run_id
    target_native_ticks = $State.target_native_ticks
    stdout = $Stdout
    stderr = $Stderr
} | ConvertTo-Json
