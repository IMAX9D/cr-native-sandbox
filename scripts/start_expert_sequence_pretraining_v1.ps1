param(
    [string]$AcceptedManifest = 'D:\AI_data\cr-native-core\expert-v1\training-dataset\version-window-20260804\accepted-cycle-clean.jsonl',
    [string]$DatasetRoot = 'D:\AI_data\cr-native-core\expert-v1\compiled\sequence-only-bc-v1',
    [string]$OutputRoot = 'D:\AI_data\cr-native-core\expert-v1\sequence-runs',
    [int]$MinimumBattles = 100000,
    [switch]$SkipCompile,
    [switch]$Smoke
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

if (-not (Test-Path -LiteralPath $AcceptedManifest)) {
    throw "Accepted expert manifest is missing: $AcceptedManifest"
}
$battleCount = 0
foreach ($line in [System.IO.File]::ReadLines($AcceptedManifest)) {
    if (-not [String]::IsNullOrWhiteSpace($line)) {
        $battleCount += 1
    }
}
if (-not $Smoke -and $battleCount -lt $MinimumBattles) {
    throw "Expert corpus is not ready: $battleCount / $MinimumBattles accepted battles"
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
if ($Smoke) {
    $DatasetRoot = 'D:\AI_data\cr-native-core\expert-v1\compiled\sequence-only-one-click-smoke-v1'
    $OutputRoot = 'D:\AI_data\cr-native-core\expert-v1\runs-smoke'
}

if (-not $SkipCompile) {
    $compileArgs = @(
        '-m', 'expert_v1.compile_sequence_dataset',
        '--accepted-manifest', $AcceptedManifest,
        '--output-root', $DatasetRoot,
        '--replace'
    )
    if ($Smoke) {
        $compileArgs += @(
            '--limit-battles', '300',
            '--validation-fraction', '0.20',
            '--test-fraction', '0.20',
            '--sequences-per-shard', '64',
            '--progress-every', '100'
        )
    }
    & $python @compileArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Sequence-only dataset compilation exited with code $LASTEXITCODE"
    }
    if (-not $Smoke) {
        $compileResult = Get-Content -LiteralPath (Join-Path $DatasetRoot 'compile-result.json') -Raw | ConvertFrom-Json
        if ([int]$compileResult.compiled_battles -lt $MinimumBattles) {
            throw "Compiled expert corpus is below threshold: $($compileResult.compiled_battles) / $MinimumBattles battles"
        }
    }
}

$manifest = Join-Path $DatasetRoot 'manifest.json'
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "Compiled sequence-only dataset is missing: $manifest"
}
$trainArgs = @(
    '-m', 'expert_v1.training_v1.train',
    '--dataset-root', $DatasetRoot,
    '--output-root', $OutputRoot,
    '--expected-source-manifest', $AcceptedManifest,
    '--run-id', "expert-sequence-v1-$stamp"
)
if ($Smoke) {
    $trainArgs += @(
        '--epochs', '1',
        '--batch-size', '2',
        '--sequence-length', '16',
        '--burn-in', '4',
        '--workers', '0',
        '--max-train-batches', '2',
        '--max-eval-batches', '1',
        '--hidden-size', '64',
        '--card-embedding-size', '32',
        '--device', 'cpu'
    )
}
& $python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Expert sequence pretraining exited with code $LASTEXITCODE"
}
