param(
    [switch]$Status,
    [switch]$Smoke,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$bundledPython = 'D:\AI_data\runtime\venv\Scripts\python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $python = $bundledPython
} else {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}

Set-Location -LiteralPath $projectRoot
$arguments = @('-m', 'expert_v1.one_click_v1')
if ($Status) { $arguments += '--status' }
if ($Smoke) { $arguments += '--smoke' }
if ($ExtraArgs) { $arguments += $ExtraArgs }

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Expert one-click v1 exited with code $LASTEXITCODE. Artifacts and stage state were preserved."
}
