$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = 'D:\AI_data\runtime\venv\Scripts\python.exe'
$dataRoot = 'D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3-current-frontier-v5'
$logRoot = Join-Path $dataRoot 'logs'
$receipt = Join-Path $dataRoot 'receipts\detached-native-12w-pipeline.json'
$workers = 12
$avds = 3
$workersPerAvd = 4
$coresPerAvd = 4
$memoryMbPerAvd = 5120
$ports = 38031..38042 | ForEach-Object { [string]$_ }
New-Item -ItemType Directory -Path (Split-Path -Parent $receipt) -Force | Out-Null
Set-Location -LiteralPath $projectRoot

$state = [ordered]@{
    schema_version = 1
    kind = 'cr_detached_native_generation_pipeline_v1'
    launcher_pid = $PID
    started_utc = [DateTime]::UtcNow.ToString('o')
    workers = $workers
    avds = $avds
    workers_per_avd = $workersPerAvd
    cores_per_avd = $coresPerAvd
    memory_mb_per_avd = $memoryMbPerAvd
    stage = 'starting_workers'
    worker_start_exit_code = $null
    generator_exit_code = $null
    worker_stop_exit_code = $null
    one_click_exit_code = $null
}
function Save-State {
    $state.updated_utc = [DateTime]::UtcNow.ToString('o')
    $state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $receipt -Encoding utf8
}
Save-State

& $python -m native_core.worker start --workers $workers --avds $avds `
    --workers-per-avd $workersPerAvd --cores-per-avd $coresPerAvd `
    --memory-mb-per-avd $memoryMbPerAvd --transport direct `
    *> (Join-Path $logRoot 'detached-native-12w-workers-start.log')
$state.worker_start_exit_code = $LASTEXITCODE
if ($LASTEXITCODE -ne 0) {
    $state.stage = 'worker_start_failed'
    Save-State
    exit $state.worker_start_exit_code
}

$state.stage = 'generating'
Save-State
$generatorArgs = @(
    (Join-Path $projectRoot 'scripts\generate_expert_native_ticks.py'), 'run',
    '--queue', (Join-Path $dataRoot 'eligibility\native-eligibility-v1\queues\authoritative-native-full.jsonl'),
    '--output-root', (Join-Path $dataRoot 'native-authoritative-ticks-v1'),
    '--template', (Join-Path $projectRoot 'examples\eight-card-bootstrap.json'),
    '--native-contract', 'D:\AI_data\cr-native-core\expert-v1\contracts\native-ingest-v150535029.json',
    '--workers', [string]$workers, '--ports'
) + $ports + @('--selection-seed', 'authoritative-schema5-v3-100k-v1')
& $python @generatorArgs `
    1> (Join-Path $logRoot 'detached-native-12w-generator.stdout.log') `
    2> (Join-Path $logRoot 'detached-native-12w-generator.stderr.log')
$state.generator_exit_code = $LASTEXITCODE

$state.stage = 'stopping_workers'
Save-State
& $python -m native_core.worker stop --workers $workers --avds $avds `
    --workers-per-avd $workersPerAvd --transport direct --stop-vm `
    *> (Join-Path $logRoot 'detached-native-12w-workers-stop.log')
$state.worker_stop_exit_code = $LASTEXITCODE
if ($state.generator_exit_code -ne 0 -or $state.worker_stop_exit_code -ne 0) {
    $state.stage = 'failed_preserved'
    Save-State
    if ($state.generator_exit_code -ne 0) { exit $state.generator_exit_code }
    exit $state.worker_stop_exit_code
}

$state.stage = 'resuming_one_click'
Save-State
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File (Join-Path $PSScriptRoot 'start_expert_one_click_v1.ps1') `
    1> (Join-Path $logRoot 'detached-post-native-one-click.stdout.log') `
    2> (Join-Path $logRoot 'detached-post-native-one-click.stderr.log')
$state.one_click_exit_code = $LASTEXITCODE
$state.stage = if ($LASTEXITCODE -eq 0) { 'complete' } else { 'one_click_failed_preserved' }
Save-State
exit $state.one_click_exit_code
