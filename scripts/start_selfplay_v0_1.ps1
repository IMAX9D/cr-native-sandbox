param(
    [ValidateSet(1000000, 5000000, 10000000)]
    [int]$TargetNativeTicks = 1000000,
    [ValidateRange(1, 256)] [int]$PairedSeeds = 16,
    [string]$RunId = "",
    [string]$Resume = "",
    [string]$DataRoot = "D:\AI_data\cr-native-core\selfplay-v0.1",
    [string]$Python = "D:\AI_data\runtime\venv\Scripts\python.exe",
    [switch]$SkipBuild,
    [switch]$SkipEvaluation,
    [switch]$KeepVms
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python runtime not found: $Python"
}

function Invoke-BuildScript {
    param([string]$Path, [string]$Name)
    $Process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Path
    ) -Wait -PassThru -WindowStyle Hidden
    if ($Process.ExitCode -ne 0) {
        throw "$Name build failed: $($Process.ExitCode)"
    }
}

if (-not $SkipBuild) {
    Invoke-BuildScript (Join-Path $PSScriptRoot "build_probe.ps1") "Java host"
    Invoke-BuildScript (Join-Path $PSScriptRoot "build_bridge.ps1") "native bridge"
}

$Arguments = @(
    (Join-Path $PSScriptRoot "run_selfplay_v0_1.py"),
    "--target-native-ticks", "$TargetNativeTicks",
    "--paired-seeds", "$PairedSeeds",
    "--data-root", $DataRoot,
    "--python", $Python
)
if ($RunId) { $Arguments += @("--run-id", $RunId) }
if ($Resume) { $Arguments += @("--resume", $Resume) }
if ($SkipEvaluation) { $Arguments += "--skip-evaluation" }
if ($KeepVms) { $Arguments += "--keep-vms" }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Self-Play v0.1 stage failed with exit code $LASTEXITCODE"
}
