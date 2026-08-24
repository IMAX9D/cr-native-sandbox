param(
    [ValidateSet("probe-baseline", "probe-detach-surface", "probe-null-surface", "probe-no-surface", "probe-create-only", "probe-minimal", "probe-direct")]
    [string]$Profile = "probe-baseline",
    [string]$Adb = $(if ($env:CR_SANDBOX_ADB) { $env:CR_SANDBOX_ADB } else { throw "Missing CR_SANDBOX_ADB; dot-source runtime.env.ps1 first" }),
    [string]$Serial = "emulator-5554",
    [string]$RuntimeDirectory = $(if ($env:CR_SANDBOX_RUNTIME_DIR) { $env:CR_SANDBOX_RUNTIME_DIR } else { throw "Missing CR_SANDBOX_RUNTIME_DIR; dot-source runtime.env.ps1 first" }),
    [string]$Bridge = "",
    [string]$BaseApk = $(if ($env:CR_SANDBOX_BASE_APK) { $env:CR_SANDBOX_BASE_APK } else { throw "Missing CR_SANDBOX_BASE_APK; dot-source runtime.env.ps1 first" }),
    [string]$ReplayJson = "",
    [string]$AssetDirectory = $(if ($env:CR_SANDBOX_ASSETS) { $env:CR_SANDBOX_ASSETS } else { throw "Missing CR_SANDBOX_ASSETS; dot-source runtime.env.ps1 first" }),
    [string]$AssetPackApk = $(if ($env:CR_SANDBOX_ASSET_PACK_APK) { $env:CR_SANDBOX_ASSET_PACK_APK } else { throw "Missing CR_SANDBOX_ASSET_PACK_APK; dot-source runtime.env.ps1 first" }),
    [string]$RemoteRoot = "/data/local/tmp/cr-native-sandbox-probe",
    [string]$EvidenceRoot = $(if ($env:CR_SANDBOX_DATA) { Join-Path $env:CR_SANDBOX_DATA "probe" } else { throw "Missing CR_SANDBOX_DATA; dot-source runtime.env.ps1 first" }),
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ReplayJson) {
    $ReplayJson = Join-Path $ProjectRoot "examples\eight-card-bootstrap.json"
}
$Jar = Join-Path $ProjectRoot "artifacts\lifecycle-probe.jar"
if (-not $Bridge) {
    $Bridge = Join-Path $ProjectRoot "artifacts\libnative_core_probe.so"
}
foreach ($Path in @($Adb, $Jar, $Bridge, $BaseApk, $AssetPackApk, $ReplayJson, (Join-Path $RuntimeDirectory "libg.so"))) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing probe input: $Path"
    }
}
if (-not (Test-Path -LiteralPath $AssetDirectory -PathType Container)) {
    throw "Missing probe asset directory: $AssetDirectory"
}

function Invoke-Adb {
    param([string[]]$CommandArguments)
    & $Adb (@("-s", $Serial) + @($CommandArguments))
    if ($LASTEXITCODE -ne 0) { throw "adb failed: $($CommandArguments -join ' ')" }
}

function Get-RemoteSha256 {
    param([string]$RemotePath)
    $Value = & $Adb -s $Serial shell "sha256sum '$RemotePath' 2>/dev/null || true"
    if ($LASTEXITCODE -ne 0) { return "" }
    $Text = ($Value | Out-String).Trim()
    if (-not $Text) { return "" }
    return (($Text -split "\s+")[0]).ToLowerInvariant()
}

function Push-Verified {
    param([string]$LocalPath, [string]$RemotePath)
    $Hash = (Get-FileHash -LiteralPath $LocalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-RemoteSha256 -RemotePath $RemotePath) -eq $Hash) { return }
    Invoke-Adb -CommandArguments @("push", $LocalPath, "$RemotePath.upload")
    if ((Get-RemoteSha256 -RemotePath "$RemotePath.upload") -ne $Hash) {
        throw "Remote hash mismatch: $RemotePath.upload"
    }
    Invoke-Adb -CommandArguments @("shell", "mv -f '$RemotePath.upload' '$RemotePath' && chmod 0666 '$RemotePath'")
}

Invoke-Adb -CommandArguments @("shell", "mkdir -p '$RemoteRoot'")
foreach ($Library in Get-ChildItem -LiteralPath $RuntimeDirectory -Filter "*.so") {
    Push-Verified -LocalPath $Library.FullName -RemotePath "$RemoteRoot/$($Library.Name)"
}
Push-Verified -LocalPath $Jar -RemotePath "$RemoteRoot/lifecycle-probe.jar"
Push-Verified -LocalPath $Bridge -RemotePath "$RemoteRoot/libnative_host_bridge.so"
Push-Verified -LocalPath $BaseApk -RemotePath "$RemoteRoot/base.apk"
Push-Verified -LocalPath $ReplayJson -RemotePath "$RemoteRoot/input-replay.json"
$AssetArchive = Join-Path $ProjectRoot "artifacts\runtime-assets.tar"
& tar.exe -cf $AssetArchive -C $AssetDirectory .
if ($LASTEXITCODE -ne 0) { throw "Failed to package runtime assets" }
$AssetOverlay = Join-Path $EvidenceRoot "runtime-assets-overlay"
New-Item -ItemType Directory -Path $AssetOverlay -Force | Out-Null
& tar.exe -xf $AssetPackApk -C $AssetOverlay --strip-components 1 `
    "assets/locations/training_arena.csv" "assets/tilemaps/tilemap.csv"
if ($LASTEXITCODE -ne 0) { throw "Failed to extract native battle-map assets" }
& tar.exe -rf $AssetArchive -C $AssetOverlay `
    "locations/training_arena.csv" "tilemaps/tilemap.csv"
if ($LASTEXITCODE -ne 0) { throw "Failed to append native battle-map assets" }
Push-Verified -LocalPath $AssetArchive -RemotePath "$RemoteRoot/runtime-assets.tar"
Invoke-Adb -CommandArguments @(
    "shell",
    "mkdir -p '$RemoteRoot/assets' && tar -xf '$RemoteRoot/runtime-assets.tar' -C '$RemoteRoot/assets'"
)

New-Item -ItemType Directory -Path $EvidenceRoot -Force | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$LogPath = Join-Path $EvidenceRoot "$Stamp-$Profile.log"
$ClassPath = "$RemoteRoot/lifecycle-probe.jar`:$RemoteRoot/base.apk"
$Launch = "cd '$RemoteRoot' && exec env CLASSPATH='$ClassPath' LD_LIBRARY_PATH='$RemoteRoot' app_process /system/bin royale.nativehost.JniHost '$RemoteRoot' '$Profile' '$RemoteRoot/input-replay.json'"
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$Output = & $Adb -s $Serial shell "$Launch; status=`$?; echo __PROBE_EXIT__=`$status; exit `$status" 2>&1
$ExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
$Output | Set-Content -LiteralPath $LogPath -Encoding utf8
if (-not $Quiet) { $Output | Out-Host }
[pscustomobject]@{
    profile = $Profile
    exit_code = $ExitCode
    log = $LogPath
    jar_sha256 = (Get-FileHash -LiteralPath $Jar -Algorithm SHA256).Hash.ToLowerInvariant()
    libg_sha256 = (Get-FileHash -LiteralPath (Join-Path $RuntimeDirectory "libg.so") -Algorithm SHA256).Hash.ToLowerInvariant()
    bridge_sha256 = (Get-FileHash -LiteralPath $Bridge -Algorithm SHA256).Hash.ToLowerInvariant()
    replay_sha256 = (Get-FileHash -LiteralPath $ReplayJson -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json | Tee-Object -FilePath "$LogPath.manifest.json"
if ($ExitCode -ne 0) { exit $ExitCode }
