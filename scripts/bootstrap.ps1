param(
    [switch]$SkipSdkInstall,
    [switch]$SkipAvd,
    [switch]$SkipRuntime
)

# One-shot environment bootstrap: installs the Android SDK packages, creates the
# rootable AVD, extracts the runtime libraries/assets and freezes the manifest.
# Idempotent; safe to re-run. Run once after `git clone` + `runtime.env.ps1`.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Require-Env {
    param([string]$Name)
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "Missing required environment variable: $Name. Copy runtime.env.example.ps1 to runtime.env.ps1, edit the paths, then run `. .\runtime.env.ps1` first."
    }
    return $Value
}

$SdkRoot = Require-Env "CR_SANDBOX_ANDROID_SDK"
$ToolsRoot = Require-Env "CR_SANDBOX_ANDROID_TOOLS"
$AvdHome = Require-Env "CR_SANDBOX_AVD_HOME"
$AvdName = if ($env:CR_SANDBOX_AVD_NAME) { $env:CR_SANDBOX_AVD_NAME } else { "royale_worker_api31" }
$SystemImage = if ($env:CR_SANDBOX_SYSTEM_IMAGE) { $env:CR_SANDBOX_SYSTEM_IMAGE } else { "system-images;android-31;default;x86_64" }

$SdkManager = Join-Path $ToolsRoot "bin\sdkmanager.bat"
$AvdManager = Join-Path $ToolsRoot "bin\avdmanager.bat"

function Invoke-SdkManager {
    param([string[]]$Arguments)
    & $SdkManager @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "sdkmanager failed with exit ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

# --- 1. Android SDK packages -------------------------------------------
if (-not $SkipSdkInstall) {
    if (-not (Test-Path -LiteralPath $SdkManager -PathType Leaf)) {
        throw "sdkmanager.bat not found: $SdkManager. Install Android command-line tools into CR_SANDBOX_ANDROID_TOOLS first (https://developer.android.com/studio#command-line-tools-only)."
    }
    # Accept licenses (feed 'y' repeatedly; sdkmanager reads lines from stdin).
    $Yes = ("y`n" * 32)
    $Yes | & $SdkManager --licenses
    Invoke-SdkManager @(
        "platform-tools",
        "emulator",
        "platforms;android-35",
        "system-images;android-31;default;x86_64",
        "ndk;27.3.13750724",
        "build-tools;35.0.0"
    )
    Write-Host "PASS sdk packages"
}

# --- 2. Rootable AVD ----------------------------------------------------
if (-not $SkipAvd) {
    if (-not (Test-Path -LiteralPath $AvdManager -PathType Leaf)) {
        throw "avdmanager.bat not found: $AvdManager"
    }
    New-Item -ItemType Directory -Path $AvdHome -Force | Out-Null
    $AvdIni = Join-Path $AvdHome "$AvdName.ini"
    if (-not (Test-Path -LiteralPath $AvdIni -PathType Leaf)) {
        # Default (AOSP) image only: Play Store images are not rootable and are
        # rejected by the sandbox host.
        "no" | & $AvdManager create avd -n $AvdName -k $SystemImage -d pixel_2 --force
        if ($LASTEXITCODE -ne 0) { throw "avdmanager create failed with exit $LASTEXITCODE" }
    }
    $ConfigIni = Join-Path $AvdHome "$AvdName.avd\config.ini"
    if (-not (Test-Path -LiteralPath $ConfigIni -PathType Leaf)) {
        throw "AVD config.ini not found after creation: $ConfigIni"
    }
    # Pin the verified baseline: 4 vCPU, 4 GB RAM, 10 GB data partition.
    $Config = Get-Content -LiteralPath $ConfigIni -Raw
    $Lines = @($Config -split "`r?`n")
    function Set-IniValue {
        param([string]$Key, [string]$Value)
        $script:Lines = @($script:Lines | Where-Object { $_ -notmatch "^$([regex]::Escape($Key))\s*=" })
        $script:Lines += "$Key=$Value"
    }
    Set-IniValue "hw.cpu.ncore" "4"
    Set-IniValue "hw.ramSize" "4096"
    Set-IniValue "disk.dataPartition.size" "10G"
    ($Lines -join "`r`n") | Set-Content -LiteralPath $ConfigIni -Encoding ascii
    Write-Host "PASS avd $AvdName ($SystemImage, 4 vCPU / 4 GB RAM / 10 GB data)"
}

# --- 3. Runtime extraction + freeze -------------------------------------
if (-not $SkipRuntime) {
    & (Join-Path $PSScriptRoot "prepare_runtime.ps1") | Out-Host
    & (Join-Path $PSScriptRoot "freeze_runtime.ps1") | Out-Host
}

Write-Host "bootstrap complete. Next: .\scripts\doctor.ps1"
