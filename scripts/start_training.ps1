param(
    [switch]$Smoke,
    [ValidateRange(1, 8)] [int]$Workers = 2,
    [ValidateRange(1, 1000000000)] [int]$Iterations = 1000000,
    [ValidateRange(1, 1024)] [int]$EpisodesPerIteration = 4,
    [ValidateRange(101, 20000)] [int]$MaxTicks = 7200,
    [int]$Seed = 1,
    [string]$Device = "auto",
    [string]$DataRoot = "D:\AI_data\cr-native-core\training",
    [string]$Python = "D:\AI_data\runtime\venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $ResolvedPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -eq $ResolvedPython) {
        throw "Python runtime not found. Expected $Python"
    }
    $Python = $ResolvedPython.Source
}

$LogRoot = Join-Path $DataRoot "launcher-logs"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Log = Join-Path $LogRoot "$Stamp-training-launch.log"

function Invoke-BuildScript {
    param([string]$Path, [string]$Name)
    $Stdout = Join-Path $LogRoot "$Stamp-$Name-stdout.log"
    $Stderr = Join-Path $LogRoot "$Stamp-$Name-stderr.log"
    $Process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Path
    ) -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    foreach ($Output in @($Stdout, $Stderr)) {
        if ((Test-Path -LiteralPath $Output) -and (Get-Item $Output).Length -gt 0) {
            Get-Content -LiteralPath $Output |
                Tee-Object -FilePath $Log -Append | Out-Host
        }
    }
    if ($Process.ExitCode -ne 0) {
        throw "$Name build failed: $($Process.ExitCode)"
    }
}

"[$(Get-Date -Format o)] building Java host" | Tee-Object -FilePath $Log -Append
Invoke-BuildScript (Join-Path $PSScriptRoot "build_probe.ps1") "java-host"

"[$(Get-Date -Format o)] building native bridge" | Tee-Object -FilePath $Log -Append
Invoke-BuildScript (Join-Path $PSScriptRoot "build_bridge.ps1") "native-bridge"

$Arguments = @(
    "-m", "training.train",
    "--workers", "$Workers",
    "--iterations", "$Iterations",
    "--episodes-per-iteration", "$EpisodesPerIteration",
    "--max-ticks", "$MaxTicks",
    "--seed", "$Seed",
    "--device", $Device,
    "--data-root", $DataRoot
)
if ($Smoke) { $Arguments += "--smoke" }

"[$(Get-Date -Format o)] starting persistent native training" |
    Tee-Object -FilePath $Log -Append
$PreviousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | Out-Host
$ExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorAction
if ($ExitCode -ne 0) {
    throw "Training exited with code $ExitCode. Log: $Log"
}
"[$(Get-Date -Format o)] training command completed" |
    Tee-Object -FilePath $Log -Append
