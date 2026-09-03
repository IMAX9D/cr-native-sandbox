[CmdletBinding()]
param(
    [string]$Checkpoint = 'D:\AI_data\cr-native-core\expert-v1\downloaded\lr-ab-20260831\candidate-lr5e-5-step157674-fp16.pt',
    [string]$Deck = '',
    [double]$PlayRateScale = 1.0,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Deck) {
    $Deck = Join-Path $Root 'examples\user-selected-heavy-control.json'
}
& (Join-Path $PSScriptRoot 'build_mumu_live_private.ps1') | Out-Host
$Python = 'D:\AI_data\runtime\venv\Scripts\python.exe'
$Arguments = @(
    '-m', 'native_core.mumu_live_controller',
    '--checkpoint', $Checkpoint,
    '--deck', $Deck,
    '--play-rate-scale', $PlayRateScale.ToString([Globalization.CultureInfo]::InvariantCulture)
)
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
