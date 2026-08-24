param(
    [string]$ExpectedHash = "96598dc9028e1802",
    [long]$ExpectedVersionCode = 150535029,
    [string]$Serial = "emulator-5554",
    [int]$Port = 37031,
    [int]$Workers = 1,
    [switch]$KeepRunning
)

# One-command smoke: build -> start (boot AVD + install + service) -> root/access
# -> package versionCode -> probe-direct (no-Surface, tick 0->100, six towers,
# public-observe-v6, state hash) -> stop. Emits the nine canonical PASS lines.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Require-Env {
    param([string]$Name)
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "Missing required environment variable: $Name. Dot-source runtime.env.ps1 first."
    }
    return $Value
}

$Adb = Require-Env "CR_SANDBOX_ADB"
$DataRoot = Require-Env "CR_SANDBOX_DATA"
$Python = if (Get-Command python -ErrorAction SilentlyContinue) { (Get-Command python).Source } else { "python" }

$Checks = [ordered]@{}
function Set-Check {
    param([string]$Name, [bool]$Pass, [string]$Detail = "")
    $Checks[$Name] = [pscustomobject]@{ pass = $Pass; detail = $Detail }
}

function Invoke-Adb {
    param([string[]]$Args)
    & $Adb (@("-s", $Serial) + @($Args)) 2>&1
}

# --- 1. doctor (toolchain + runtime hashes) ------------------------------
$DoctorJson = & (Join-Path $PSScriptRoot "doctor.ps1") -Json | Out-String
$Doctor = @($DoctorJson | ConvertFrom-Json)
$DoctorExit = $LASTEXITCODE
$ToolchainNames = @("environment","python 3.11+","adb","emulator","sdkmanager","avdmanager","android.jar (platform 35)","r8.jar","javac","clang++ (NDK r27d)","avd home")
$ToolchainOk = $DoctorExit -eq 0
$RuntimeHashes = $Doctor | Where-Object { $_.name -eq "runtime hashes" } | Select-Object -First 1
$RuntimeHashesOk = ($null -ne $RuntimeHashes) -and $RuntimeHashes.pass
Set-Check "toolchain" $ToolchainOk
Set-Check "runtime hashes" $RuntimeHashesOk

# --- 2. build ------------------------------------------------------------
& (Join-Path $PSScriptRoot "build_probe.ps1") | Out-Host
& (Join-Path $PSScriptRoot "build_bridge.ps1") | Out-Host

# --- 3. boot AVD + install package + start service -----------------------
& $Python -m native_core.worker start --workers $Workers --transport adb `
    --base-port $Port --avd-name $(if ($env:CR_SANDBOX_AVD_NAME) { $env:CR_SANDBOX_AVD_NAME } else { "royale_worker_api31" }) `
    | Out-Host
if ($LASTEXITCODE -ne 0) { throw "worker start failed with exit $LASTEXITCODE" }

# --- 4. AVD root / access ------------------------------------------------
Invoke-Adb @("root") | Out-Null
Start-Sleep -Seconds 1
$Id = (Invoke-Adb @("shell", "id") | Out-String).Trim()
$RootOk = $Id -match "uid=0\(root\)"
Set-Check "AVD root/access" $RootOk $Id

# --- 5. package versionCode ----------------------------------------------
$Pkg = (Invoke-Adb @("shell", "dumpsys", "package", "com.supercell.clashroyale") | Out-String)
$VersionOk = $Pkg -match "versionCode=$ExpectedVersionCode"
Set-Check "package versionCode=$ExpectedVersionCode" $VersionOk

# --- 6. probe-direct: no-Surface, tick 0->100, six towers, hash, scope ---
$ProbeManifest = (& (Join-Path $PSScriptRoot "run_probe.ps1") -Profile probe-direct -Quiet `
    -Serial $Serial -RemoteRoot "/data/local/tmp/cr-native-smoke" | Out-String | ConvertFrom-Json)
$LogPath = $ProbeManifest.log
if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
    throw "smoke probe did not produce a log: $LogPath"
}
$CompleteLine = Get-Content -LiteralPath $LogPath | Where-Object {
    $_ -like '*"stage":"probe_result"*' -and $_ -like '*"event":"complete"*'
} | Select-Object -Last 1
if (-not $CompleteLine) { throw "smoke probe has no complete probe_result" }
$Value = ($CompleteLine | ConvertFrom-Json).value

$HasSurface = [bool](Select-String -LiteralPath $LogPath -SimpleMatch '"stage":"surface_create"' -Quiet)
$NoSurface = -not $HasSurface
Set-Check "no-Surface host" $NoSurface

$Towers = @($Value.state.episode.crown_towers)
$SixTowers = ([int]$Value.state.entity_count -eq 6 -and $Towers.Count -eq 6)
Set-Check "six towers" $SixTowers ("entity_count=" + $Value.state.entity_count)

$TickOk = ([int]$Value.step.tick_before -eq 0 -and [int]$Value.step.tick_after -eq 100)
Set-Check "tick 0 -> 100" $TickOk ("before=" + $Value.step.tick_before + " after=" + $Value.step.tick_after)

$Scope = [string]$Value.state.state_hash_scope
$ScopeOk = $Scope -eq "public-observe-v6"
Set-Check "public-observe-v6" $ScopeOk $Scope

$Hash = [string]$Value.state.state_hash
$HashOk = $Hash -eq $ExpectedHash
Set-Check "hash $ExpectedHash" $HashOk $Hash

# --- 7. stop -------------------------------------------------------------
if (-not $KeepRunning) {
    & $Python -m native_core.worker stop --workers $Workers --base-port $Port --stop-vm | Out-Host
}

# --- consolidated nine-line summary --------------------------------------
$Summary = [ordered]@{
    toolchain = $Checks["toolchain"].pass
    runtime_hashes = $Checks["runtime hashes"].pass
    avd_root = $Checks["AVD root/access"].pass
    package_version = $Checks["package versionCode=$ExpectedVersionCode"].pass
    no_surface = $Checks["no-Surface host"].pass
    six_towers = $Checks["six towers"].pass
    tick = $Checks["tick 0 -> 100"].pass
    scope = $Checks["public-observe-v6"].pass
    hash = $Checks["hash $ExpectedHash"].pass
}

Write-Host ""
Write-Host $(if ($Summary.toolchain) { "PASS toolchain" } else { "FAIL toolchain" })
Write-Host $(if ($Summary.runtime_hashes) { "PASS runtime hashes" } else { "FAIL runtime hashes" })
Write-Host $(if ($Summary.avd_root) { "PASS AVD root/access" } else { "FAIL AVD root/access" })
Write-Host $(if ($Summary.package_version) { "PASS package versionCode=$ExpectedVersionCode" } else { "FAIL package versionCode=$ExpectedVersionCode" })
Write-Host $(if ($Summary.no_surface) { "PASS no-Surface host" } else { "FAIL no-Surface host" })
Write-Host $(if ($Summary.six_towers) { "PASS six towers" } else { "FAIL six towers" })
Write-Host $(if ($Summary.tick) { "PASS tick 0 -> 100" } else { "FAIL tick 0 -> 100" })
Write-Host $(if ($Summary.scope) { "PASS public-observe-v6" } else { "FAIL public-observe-v6" })
Write-Host $(if ($Summary.hash) { "PASS hash $ExpectedHash" } else { "FAIL hash $ExpectedHash" })

$AllOk = @($Summary.Values) -notcontains $false
if (-not $AllOk) { throw "smoke test failed" }
exit 0
