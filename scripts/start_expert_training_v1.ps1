param(
    [switch]$Smoke,
    [string]$DatasetRoot = 'D:\AI_data\cr-native-core\expert-v1\compiled\native-bc-v1',
    [string]$OutputRoot = 'D:\AI_data\cr-native-core\expert-v1\runs'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$bundledPython = 'D:\AI_data\runtime\venv\Scripts\python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $python = $bundledPython
} else {
    $command = Get-Command python.exe -ErrorAction Stop
    $python = $command.Source
}

Set-Location -LiteralPath $projectRoot
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
if ($Smoke) {
    $DatasetRoot = 'D:\AI_data\cr-native-core\expert-v1\compiled\smoke-native-bc-v1'
    $OutputRoot = 'D:\AI_data\cr-native-core\expert-v1\runs-smoke'
    & $python -m expert_v1.training_v1.train `
        --smoke `
        --dataset-root $DatasetRoot `
        --output-root $OutputRoot `
        --run-id "expert-v1-smoke-$stamp" `
        --epochs 1 `
        --batch-size 2 `
        --sequence-length 16 `
        --burn-in 4 `
        --workers 0 `
        --max-train-batches 2 `
        --max-eval-batches 1 `
        --hidden-size 64 `
        --card-embedding-size 32 `
        --device cpu
} else {
    $manifest = Join-Path $DatasetRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "Compiled expert dataset is not ready: $manifest"
    }
    & $python -m expert_v1.training_v1.train `
        --dataset-root $DatasetRoot `
        --output-root $OutputRoot `
        --run-id "expert-v1-$stamp"
}
if ($LASTEXITCODE -ne 0) {
    throw "Expert training exited with code $LASTEXITCODE"
}

