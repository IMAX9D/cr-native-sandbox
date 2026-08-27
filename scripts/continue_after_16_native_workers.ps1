param(
    [Parameter(Mandatory = $true)]
    [int]$GeneratorPid
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = 'D:\AI_data\runtime\venv\Scripts\python.exe'
$dataRoot = 'D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3-current-frontier-v5'
$logRoot = Join-Path $dataRoot 'logs'
$receipt = Join-Path $dataRoot 'receipts\native-16w-continuation.json'
New-Item -ItemType Directory -Path (Split-Path -Parent $receipt) -Force | Out-Null

$generator = Get-Process -Id $GeneratorPid -ErrorAction Stop
$generator.WaitForExit()
$exitCode = $generator.ExitCode

Set-Location -LiteralPath $projectRoot
& $python -m native_core.worker stop --workers 16 --avds 4 --workers-per-avd 4 --transport direct --stop-vm |
    Out-File -LiteralPath (Join-Path $logRoot 'native-16w-stop.log') -Encoding utf8
$stopExitCode = $LASTEXITCODE

$payload = [ordered]@{
    schema_version = 1
    kind = 'cr_native_16w_continuation_v1'
    completed_utc = [DateTime]::UtcNow.ToString('o')
    generator_pid = $GeneratorPid
    generator_exit_code = $exitCode
    worker_stop_exit_code = $stopExitCode
    one_click_resumed = $false
}

if ($exitCode -eq 0 -and $stopExitCode -eq 0) {
    $launcher = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', (Join-Path $PSScriptRoot 'start_expert_one_click_v1.ps1')
    ) -WorkingDirectory $projectRoot -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $logRoot 'post-16w-one-click.stdout.log') `
      -RedirectStandardError (Join-Path $logRoot 'post-16w-one-click.stderr.log') `
      -PassThru
    $payload.one_click_resumed = $true
    $payload.one_click_launcher_pid = $launcher.Id
}

$payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $receipt -Encoding utf8
if ($exitCode -ne 0) {
    exit $exitCode
}
if ($stopExitCode -ne 0) {
    exit $stopExitCode
}
