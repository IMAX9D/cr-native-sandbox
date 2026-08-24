param(
    [switch]$Json
)

# Preflight check: reports every dependency before the sandbox starts, instead of
# failing one file at a time. Exits 0 only when every hard check passes.

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Get-Env {
    param([string]$Name)
    [Environment]::GetEnvironmentVariable($Name)
}

$Checks = New-Object System.Collections.Generic.List[object]
function Add-Check {
    param([string]$Name, [bool]$Pass, [string]$Detail = "")
    $Checks.Add([pscustomobject]@{ name = $Name; pass = $Pass; detail = $Detail })
    if ($Json) { return }
    if ($Pass) { Write-Host "PASS $Name" }
    else { Write-Host "FAIL $Name $Detail" }
}

function Test-EnvConfigured {
    $Required = @(
        "CR_SANDBOX_ANDROID_SDK", "CR_SANDBOX_ADB", "CR_SANDBOX_ANDROID_TOOLS",
        "CR_SANDBOX_ANDROID_JAR", "CR_SANDBOX_NDK", "CR_SANDBOX_JDK",
        "CR_SANDBOX_AVD_HOME", "CR_SANDBOX_APKS", "CR_SANDBOX_RUNTIME_DIR",
        "CR_SANDBOX_BASE_APK", "CR_SANDBOX_ASSET_PACK_APK", "CR_SANDBOX_ASSETS",
        "CR_SANDBOX_DATA"
    )
    $Missing = @($Required | Where-Object { [string]::IsNullOrWhiteSpace((Get-Env $_)) })
    if ($Missing.Count -gt 0) {
        Add-Check "environment" $false ("missing: " + ($Missing -join ", ") + ". Copy runtime.env.example.ps1 to runtime.env.ps1, edit and dot-source it.")
        return $false
    }
    Add-Check "environment" $true "CR_SANDBOX_* variables present"
    return $true
}

function Test-Tool {
    param([string]$Name, [string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Add-Check $Name $false "not found: $Path"
        return $false
    }
    Add-Check $Name $true $Path
    return $true
}

function Test-Dir {
    param([string]$Name, [string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        Add-Check $Name $false "not found: $Path"
        return $false
    }
    Add-Check $Name $true $Path
    return $true
}

# --- 1. environment + toolchain -----------------------------------------
$EnvOk = Test-EnvConfigured
if ($EnvOk) {
    Test-Tool "python" (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)
    $Py = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
    if ($Py) {
        $PyVer = & $Py -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
        $PyMajor = 0
        if ($PyVer -match '^(\d+)') { $PyMajor = [int]$Matches[1] }
        Add-Check "python 3.11+" ($PyMajor -ge 3 -and ([int]($PyVer -split '\.')[1] -ge 11)) "python $PyVer"
    }
    Test-Tool "adb" (Get-Env "CR_SANDBOX_ADB")
    Test-Tool "emulator" (Join-Path (Get-Env "CR_SANDBOX_ANDROID_SDK") "emulator\emulator.exe")
    Test-Tool "sdkmanager" (Join-Path (Get-Env "CR_SANDBOX_ANDROID_TOOLS") "bin\sdkmanager.bat")
    Test-Tool "avdmanager" (Join-Path (Get-Env "CR_SANDBOX_ANDROID_TOOLS") "bin\avdmanager.bat")
    Test-Tool "android.jar (platform 35)" (Get-Env "CR_SANDBOX_ANDROID_JAR")
    Test-Tool "r8.jar" (Join-Path (Get-Env "CR_SANDBOX_ANDROID_TOOLS") "lib\r8.jar")
    Test-Tool "javac" (Join-Path (Get-Env "CR_SANDBOX_JDK") "bin\javac.exe")
    Test-Tool "clang++ (NDK r27d)" (Join-Path (Get-Env "CR_SANDBOX_NDK") "toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe")
    Test-Dir "avd home" (Get-Env "CR_SANDBOX_AVD_HOME")
}

# --- 2. runtime hashes ---------------------------------------------------
$RuntimeOk = $true
if ($EnvOk) {
    $DataDir = Get-Env "CR_SANDBOX_DATA"
    $FrozenManifest = Join-Path $DataDir "manifest\runtime-manifest.json"
    $TemplateManifest = Join-Path $ProjectRoot "bindings\runtime-manifest.json"
    $ManifestPath = if (Test-Path -LiteralPath $FrozenManifest -PathType Leaf) { $FrozenManifest } else { $TemplateManifest }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

    $ApksDir = Get-Env "CR_SANDBOX_APKS"
    $LibsDir = Get-Env "CR_SANDBOX_RUNTIME_DIR"
    $Missing = New-Object System.Collections.Generic.List[string]
    $Mismatch = New-Object System.Collections.Generic.List[string]

    foreach ($Apk in $Manifest.apks) {
        $Path = Join-Path $ApksDir $Apk.name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { $Missing.Add($Apk.name); continue }
        if ($Apk.sha256) {
            $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($Hash -ne $Apk.sha256) { $Mismatch.Add("$($Apk.name) sha256") }
        }
    }
    foreach ($Lib in $Manifest.native_libs) {
        $Path = Join-Path $LibsDir $Lib.name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { $Missing.Add($Lib.name); continue }
        if ($Lib.sha256) {
            $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($Hash -ne $Lib.sha256) { $Mismatch.Add("$($Lib.name) sha256") }
        }
    }

    $LibgPath = Join-Path $LibsDir "libg.so"
    $FrozenLibg = [string]$Manifest.frozen_libg_sha256
    if (Test-Path -LiteralPath $LibgPath -PathType Leaf) {
        $LibgHash = (Get-FileHash -LiteralPath $LibgPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($LibgHash -ne $FrozenLibg) { $Mismatch.Add("libg.so (frozen)") }
    } else {
        $Missing.Add("libg.so")
    }

    $Detail = ""
    if ($Missing.Count) { $Detail += "missing: " + ($Missing -join ", ") + "; " }
    if ($Mismatch.Count) { $Detail += "hash mismatch: " + ($Mismatch -join ", ") }
    $RuntimeOk = ($Missing.Count -eq 0 -and $Mismatch.Count -eq 0)
    Add-Check "runtime hashes" $RuntimeOk $Detail.Trim()
}

# --- 3. assets -----------------------------------------------------------
if ($EnvOk) {
    $AssetsDir = Get-Env "CR_SANDBOX_ASSETS"
    $CsvClient = Test-Path -LiteralPath (Join-Path $AssetsDir "csv_client") -PathType Container
    $CsvLogic = Test-Path -LiteralPath (Join-Path $AssetsDir "csv_logic") -PathType Container
    $Arena = Test-Path -LiteralPath (Join-Path $AssetsDir "locations\training_arena.csv") -PathType Leaf
    $Tilemap = Test-Path -LiteralPath (Join-Path $AssetsDir "tilemaps\tilemap.csv") -PathType Leaf
    $CsvCount = @(Get-ChildItem -LiteralPath $AssetsDir -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue).Count
    $Ok = $CsvClient -and $CsvLogic -and $Arena -and $Tilemap
    Add-Check "runtime assets" $Ok ("csv_client=$CsvClient csv_logic=$CsvLogic arena=$Arena tilemap=$Tilemap csv_files=$CsvCount")
}

# --- 4. AVD --------------------------------------------------------------
$AvdOk = $false
if ($EnvOk) {
    $AvdHome = Get-Env "CR_SANDBOX_AVD_HOME"
    $AvdName = if (Get-Env "CR_SANDBOX_AVD_NAME") { Get-Env "CR_SANDBOX_AVD_NAME" } else { "royale_worker_api31" }
    $Ini = Test-Path -LiteralPath (Join-Path $AvdHome "$AvdName.ini") -PathType Leaf
    $Config = Test-Path -LiteralPath (Join-Path $AvdHome "$AvdName.avd\config.ini") -PathType Leaf
    $AvdOk = $Ini -and $Config
    Add-Check "avd" $AvdOk "$AvdName (run bootstrap.ps1 if missing)"
}

# --- 5. virtualization / WHPX -------------------------------------------
$VirtDetail = "unable to probe"
$VirtOk = $true
try {
    $Whpx = Get-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -ErrorAction Stop
    $VirtDetail = "WHPX state=$($Whpx.State)"
    if ($Whpx.State -ne "Enabled") { $VirtOk = $false; $VirtDetail += " (enable 'Windows Hypervisor Platform' for emulator acceleration)" }
} catch {
    # DISM requires elevation; fall back to a CPU flag probe.
    try {
        $Virt = (Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled
        $VirtDetail = "VT-x/AMD-V firmware enabled=$Virt"
        if (-not $Virt) { $VirtOk = $false; $VirtDetail += " (enable virtualization in BIOS/UEFI)" }
    } catch {
        $VirtOk = $true
        $VirtDetail = "unable to probe (run as Administrator for full check)"
    }
}
Add-Check "virtualization/WHPX" $VirtOk $VirtDetail

# --- 6. port availability ------------------------------------------------
$PortDetail = ""
$PortOk = $true
foreach ($Port in @(5554, 5555, 37031, 37032)) {
    $Listener = $null
    try {
        $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $Listener.Start()
    } catch {
        $PortOk = $false
        $PortDetail += "port $Port in use; "
    } finally {
        if ($Listener) { $Listener.Stop() }
    }
}
Add-Check "ports" $PortOk ($(if ($PortDetail) { $PortDetail.Trim() } else { "5554/5555/37031/37032 free" }))

# --- 7. disk space -------------------------------------------------------
$DiskDetail = ""
$DiskOk = $true
try {
    $DataDrive = (Split-Path -Qualifier (Get-Env "CR_SANDBOX_DATA"))[0]
    $Drive = Get-PSDrive -Name ($DataDrive.TrimEnd(':')) -ErrorAction Stop
    $FreeGb = [math]::Round($Drive.Free / 1GB, 1)
    $DiskOk = $Drive.Free -gt 30GB
    $DiskDetail = "$DataDrive free=$FreeGb GB (need ~30 GB for AVD + SDK + runtime)"
} catch {
    $DiskOk = $true
    $DiskDetail = "unable to probe free space"
}
Add-Check "disk space" $DiskOk $DiskDetail

# --- 8. execution policy ------------------------------------------------
$Policy = Get-ExecutionPolicy -Scope Process
$PolicyOk = $true
if ($Policy -eq "Restricted" -and -not $env:CR_SANDBOX_POLICY_OK) {
    # Scripts run with -ExecutionPolicy Bypass internally; this is advisory.
    $PolicyOk = $true
}
Add-Check "execution policy" $PolicyOk "process=$Policy (scripts self-bypass)"

# --- summary ------------------------------------------------------------
$Hard = @($Checks | Where-Object { $_.name -in @("environment","runtime hashes","runtime assets","avd","adb","emulator","android.jar (platform 35)","r8.jar","javac","clang++ (NDK r27d)") })
$Failed = @($Hard | Where-Object { -not $_.pass })

if ($Json) {
    @($Checks) | ConvertTo-Json -Depth 4
} elseif ($Failed.Count -eq 0) {
    Write-Host "doctor: all hard checks passed"
} else {
    Write-Host ("doctor: {0} hard check(s) failed: {1}" -f $Failed.Count, (($Failed | ForEach-Object { $_.name }) -join ", "))
}

if ($Failed.Count -eq 0) { exit 0 } else { exit 1 }
