param(
    [string]$Python = "",
    [string]$Checkpoint = "",
    [string]$Replay = "",
    [int]$Port = 37031,
    [switch]$SkipBuild,
    [switch]$RebuildHost,
    [switch]$SkipWorkerStart,
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$LocalEnvironment = Join-Path $ProjectRoot "runtime.env.ps1"
if (Test-Path -LiteralPath $LocalEnvironment) { . $LocalEnvironment }
if (-not $Python) { $Python = if ($env:CR_MATCH_PYTHON) { $env:CR_MATCH_PYTHON } else { "python" } }
if (-not $Checkpoint) { $Checkpoint = Join-Path $ProjectRoot "models\hokoff-bc-step1037042.pt" }
if (-not $Replay) { $Replay = Join-Path $ProjectRoot "examples\hog-2.6-evo-hero.json" }
if (-not $env:CR_SANDBOX_BINDERLESS_BOOT) { $env:CR_SANDBOX_BINDERLESS_BOOT = '1' }
# Check before the worker starter, which resets its bootstrap battle.
$PortPattern = '"?--port"?\s+"?' + $Port + '"?(\s|$)'
$ExistingBox = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^pythonw?\.exe$' -and
    $_.CommandLine -match '"?-m"?\s+"?hokoff_model\.match_box"?(\s|$)' -and
    $_.CommandLine -match $PortPattern
})
if ($ExistingBox.Count) {
    Write-Host "A BC match box is already open on port $Port. No reset or second window was started."
    exit 0
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    & $Python (Join-Path $PSScriptRoot "download_hokoff_match_model.py") --output $Checkpoint
    if ($LASTEXITCODE -ne 0) { throw "Pinned model download failed" }
}
if (-not $SkipBuild) {
    if ($RebuildHost) {
        & (Join-Path $PSScriptRoot "build_probe.ps1") | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Java host build failed" }
        & (Join-Path $PSScriptRoot "build_bridge.ps1") | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Native bridge build failed" }
    } else {
        & $Python (Join-Path $PSScriptRoot "prepare_hokoff_match_host.py")
        if ($LASTEXITCODE -ne 0) { throw "Prebuilt host verification failed" }
    }
}
if (-not $SkipWorkerStart) {
    & $Python -m native_core.worker start --workers 1 --base-port $Port | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Native match worker failed" }
}
$LogDirectory = Join-Path $ProjectRoot "artifacts\hokoff-match-box"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$Arguments = @("-m", "hokoff_model.match_box", "--checkpoint", $Checkpoint,
    "--replay", $Replay, "--port", "$Port", "--session-root", $LogDirectory)
if ($Smoke) {
    & $Python @Arguments --smoke
    if ($LASTEXITCODE -ne 0) { throw "Native BC integration smoke failed" }
    exit 0
}
$ResolvedPython = (Get-Command $Python -ErrorAction Stop).Source
$Pythonw = Join-Path (Split-Path -Parent $ResolvedPython) "pythonw.exe"
if (-not (Test-Path -LiteralPath $Pythonw)) { $Pythonw = $ResolvedPython }
$Stamp = (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N").Substring(0,6)
$QuotedArguments = ($Arguments | ForEach-Object { '"' + ($_ -replace '"','\"') + '"' }) -join ' '
$StartedAt = [DateTime]::UtcNow
$Process = Start-Process -FilePath $Pythonw -ArgumentList $QuotedArguments -WorkingDirectory $ProjectRoot -WindowStyle Normal -PassThru `
    -RedirectStandardOutput (Join-Path $LogDirectory "$Stamp.stdout.log") -RedirectStandardError (Join-Path $LogDirectory "$Stamp.stderr.log")
$ReadyDeadline = [DateTime]::UtcNow.AddSeconds(45)
while ([DateTime]::UtcNow -lt $ReadyDeadline) {
    $StatusPath = Join-Path $LogDirectory 'box-status.json'
    if (Test-Path -LiteralPath $StatusPath) {
        try { $BoxStatus = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json } catch { $BoxStatus = $null }
        if ($BoxStatus -and [DateTime]::Parse($BoxStatus.updated_utc).ToUniversalTime() -ge $StartedAt) {
            if ($BoxStatus.error) { throw "BC match box failed: $($BoxStatus.error)" }
            if ($BoxStatus.ready -and $BoxStatus.window_visible) {
                Write-Host "BC match box ready (PID $($Process.Id)). Click Start in the window. Logs: $LogDirectory"
                exit 0
            }
        }
    }
    if ($Process.HasExited) { throw "BC GUI process exited; inspect $LogDirectory\$Stamp.stderr.log" }
    Start-Sleep -Milliseconds 250
}
throw "BC GUI did not report ready within 45 seconds; inspect $LogDirectory"
