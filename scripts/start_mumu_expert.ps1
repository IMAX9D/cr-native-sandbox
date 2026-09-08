[CmdletBinding()]
param(
    [string]$Checkpoint = '',
    [string]$Deck = '',
    [string]$Python = '',
    [string]$NdkRoot = '',
    [string]$Adb = '',
    [string]$Serial = '127.0.0.1:16416',
    [double]$PlayRateScale = 1.0,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Checkpoint) { $Checkpoint = $env:CR_EXPERT_CHECKPOINT }
if (-not $Checkpoint) {
    $LocalLegacy = 'D:\AI_data\cr-native-core\expert-v1\downloaded\lr-ab-20260831\candidate-lr5e-5-step157674-fp16.pt'
    if (Test-Path -LiteralPath $LocalLegacy) { $Checkpoint = $LocalLegacy }
}
if (-not $Checkpoint -or -not (Test-Path -LiteralPath $Checkpoint)) { throw 'Pass -Checkpoint with a trusted compatible model, or set CR_EXPERT_CHECKPOINT.' }
if (-not $Python) { $Python = $env:CR_EXPERT_PYTHON }
if (-not $Python) {
    $LocalPython = 'D:\AI_data\runtime\venv\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $LocalPython) { $LocalPython } else { 'python' }
}
if (-not $Deck) {
    $Deck = Join-Path $Root 'examples\user-selected-heavy-control.json'
}
if ($NdkRoot -or -not (Test-Path -LiteralPath (Join-Path $Root 'artifacts\mumu-live\mumu-live-reader-v2-x86_64'))) {
    & (Join-Path $PSScriptRoot 'build_mumu_live_private.ps1') -NdkRoot $NdkRoot | Out-Host
}
$Arguments = @(
    '-m', 'native_core.mumu_live_controller',
    '--checkpoint', $Checkpoint,
    '--deck', $Deck,
    '--serial', $Serial,
    '--play-rate-scale', $PlayRateScale.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($Adb) { $Arguments += @('--adb', $Adb) }
if ($DryRun) { $Arguments += '--dry-run' }
Push-Location $Root
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "MuMu expert controller exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
