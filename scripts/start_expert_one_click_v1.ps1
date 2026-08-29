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
if (-not $Status -and -not $Smoke) {
    $defaultDataRoot = 'D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3-current-frontier-v5'
    $subsetReceipt = Join-Path $defaultDataRoot 'receipts\native-training-subset-finalized-v1.json'
    if (
        (Test-Path -LiteralPath $subsetReceipt) -and
        -not ($ExtraArgs -contains '--continue-after-native-exclusion')
    ) {
        $arguments += '--continue-after-native-exclusion'
    }
    $progressGui = Join-Path $projectRoot 'scripts\expert_compile_progress_gui.py'
    $pythonw = [System.IO.Path]::ChangeExtension($python, 'exe')
    $pythonwCandidate = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
    if (Test-Path -LiteralPath $pythonwCandidate) {
        $pythonw = $pythonwCandidate
    }
    if (Test-Path -LiteralPath $progressGui) {
        Start-Process -FilePath $pythonw -ArgumentList @($progressGui) -WorkingDirectory $projectRoot
    }
}
if ($ExtraArgs) { $arguments += $ExtraArgs }

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Expert one-click v1 exited with code $LASTEXITCODE. Artifacts and stage state were preserved."
}
