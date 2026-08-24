param(
    [string]$Adb = $(if ($env:CR_SANDBOX_ADB) { $env:CR_SANDBOX_ADB } else { "D:\Codex\toolchains\android-sdk\platform-tools\adb.exe" }),
    [string]$Serial = "emulator-5554",
    [int]$Port = 37031,
    [int]$Slot = 0,
    [string]$RuntimeDirectory = $(if ($env:CR_SANDBOX_RUNTIME_DIR) { $env:CR_SANDBOX_RUNTIME_DIR } else { "D:\Codex\E\AI ClashRoyale\native_host\build\runtime-x86_64" }),
    [string]$BaseApk = $(if ($env:CR_SANDBOX_BASE_APK) { $env:CR_SANDBOX_BASE_APK } else { "D:\Codex\E\AI ClashRoyale\runtime\installed-150535029\apks\base.apk" }),
    [string]$AssetDirectory = $(if ($env:CR_SANDBOX_ASSETS) { $env:CR_SANDBOX_ASSETS } else { "D:\Codex\E\AI ClashRoyale\runtime\installed-150535029\extracted\assets" }),
    [string]$AssetPackApk = $(if ($env:CR_SANDBOX_ASSET_PACK_APK) { $env:CR_SANDBOX_ASSET_PACK_APK } else { "D:\Codex\E\AI ClashRoyale\runtime\installed-150535029\apks\split_install_time_asset_pack.apk" }),
    [string]$BootstrapReplayJson = "",
    [string]$DataRoot = $(if ($env:CR_SANDBOX_DATA) { $env:CR_SANDBOX_DATA } else { "D:\AI_data\cr-native-sandbox" }),
    [ValidateRange(30, 900)] [int]$ReadyTimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $BootstrapReplayJson) {
    $BootstrapReplayJson = Join-Path $ProjectRoot "examples\eight-card-bootstrap.json"
}
$Jar = Join-Path $ProjectRoot "artifacts\lifecycle-probe.jar"
$Bridge = Join-Path $ProjectRoot "artifacts\libnative_core_probe.so"
$RemoteRoot = "/data/local/tmp/cr-native-direct-$Slot"
$EvidenceRoot = Join-Path $DataRoot "worker"
$AssetOverlay = Join-Path $EvidenceRoot "runtime-assets-overlay"
$AssetArchive = Join-Path $ProjectRoot "artifacts\runtime-assets.tar"

foreach ($Path in @(
    $Adb, $Jar, $Bridge, $BaseApk, $AssetPackApk,
    $BootstrapReplayJson, (Join-Path $RuntimeDirectory "libg.so")
)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing direct-worker input: $Path"
    }
}
if (-not (Test-Path -LiteralPath $AssetDirectory -PathType Container)) {
    throw "Missing runtime asset directory: $AssetDirectory"
}
New-Item -ItemType Directory -Path $EvidenceRoot -Force | Out-Null
New-Item -ItemType Directory -Path $AssetOverlay -Force | Out-Null

function Invoke-Adb {
    param([string[]]$CommandArguments, [switch]$AllowFailure)
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $Output = & $Adb (@("-s", $Serial) + @($CommandArguments)) 2>&1
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorAction
    if ($ExitCode -ne 0 -and -not $AllowFailure) {
        throw "adb failed: $($CommandArguments -join ' ')`n$($Output | Out-String)"
    }
    return $Output
}

function Get-RemoteSha256 {
    param([string]$RemotePath)
    $Value = Invoke-Adb -AllowFailure -CommandArguments @(
        "shell", "sha256sum '$RemotePath' 2>/dev/null || true"
    )
    $Text = ($Value | Out-String).Trim()
    if (-not $Text) { return "" }
    return (($Text -split "\s+")[0]).ToLowerInvariant()
}

function Push-Verified {
    param([string]$LocalPath, [string]$RemotePath)
    $Hash = (Get-FileHash -LiteralPath $LocalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-RemoteSha256 $RemotePath) -eq $Hash) { return }
    Invoke-Adb @("push", $LocalPath, "$RemotePath.upload") | Out-Null
    if ((Get-RemoteSha256 "$RemotePath.upload") -ne $Hash) {
        throw "Remote hash mismatch: $RemotePath.upload"
    }
    Invoke-Adb @(
        "shell", "mv -f '$RemotePath.upload' '$RemotePath' && chmod 0666 '$RemotePath'"
    ) | Out-Null
}

function Invoke-JsonRequest {
    param([int]$RequestPort, [hashtable]$Payload, [int]$TimeoutMs = 2000)
    $Client = [Net.Sockets.TcpClient]::new()
    try {
        $Pending = $Client.ConnectAsync("127.0.0.1", $RequestPort)
        if (-not $Pending.Wait($TimeoutMs)) { throw "connect timeout" }
        $Client.ReceiveTimeout = $TimeoutMs
        $Client.SendTimeout = $TimeoutMs
        $Stream = $Client.GetStream()
        $Writer = [IO.StreamWriter]::new($Stream, [Text.UTF8Encoding]::new($false))
        $Reader = [IO.StreamReader]::new($Stream, [Text.UTF8Encoding]::new($false))
        $Writer.NewLine = "`n"
        $Writer.WriteLine(($Payload | ConvertTo-Json -Compress))
        $Writer.Flush()
        $Line = $Reader.ReadLine()
        if (-not $Line) { throw "empty response" }
        return $Line | ConvertFrom-Json
    } finally {
        $Client.Dispose()
    }
}

Invoke-Adb @("shell", "mkdir -p '$RemoteRoot'") | Out-Null
foreach ($Library in Get-ChildItem -LiteralPath $RuntimeDirectory -Filter "*.so") {
    Push-Verified $Library.FullName "$RemoteRoot/$($Library.Name)"
}
Push-Verified $Jar "$RemoteRoot/lifecycle-probe.jar"
Push-Verified $Bridge "$RemoteRoot/libnative_host_bridge.so"
Push-Verified $BaseApk "$RemoteRoot/base.apk"
Push-Verified $BootstrapReplayJson "$RemoteRoot/bootstrap-replay.json"

& tar.exe -cf $AssetArchive -C $AssetDirectory .
if ($LASTEXITCODE -ne 0) { throw "Failed to package runtime assets" }
& tar.exe -xf $AssetPackApk -C $AssetOverlay --strip-components 1 `
    "assets/locations/training_arena.csv" "assets/tilemaps/tilemap.csv"
if ($LASTEXITCODE -ne 0) { throw "Failed to extract native battle-map assets" }
& tar.exe -rf $AssetArchive -C $AssetOverlay `
    "locations/training_arena.csv" "tilemaps/tilemap.csv"
if ($LASTEXITCODE -ne 0) { throw "Failed to append native battle-map assets" }
Push-Verified $AssetArchive "$RemoteRoot/runtime-assets.tar"
Invoke-Adb @(
    "shell",
    "mkdir -p '$RemoteRoot/assets' && tar -xf '$RemoteRoot/runtime-assets.tar' -C '$RemoteRoot/assets'"
) | Out-Null

function Get-ServicePids {
    $ProcessLines = Invoke-Adb -AllowFailure -CommandArguments @(
        "shell", "ps -A -o PID,ARGS"
    )
    return @($ProcessLines | ForEach-Object {
        $Line = [string]$_
        if ($Line.Contains("royale.nativehost.JniHost") -and
            $Line.Contains($RemoteRoot) -and
            $Line.Contains("serve-direct")) {
            $Fields = $Line.Trim() -split "\s+", 2
            if ($Fields[0] -match '^\d+$') { [int]$Fields[0] }
        }
    })
}
foreach ($OldPid in Get-ServicePids) {
    Invoke-Adb -AllowFailure -CommandArguments @(
        "shell", "kill '$OldPid' 2>/dev/null || true"
    ) | Out-Null
}
Invoke-Adb -AllowFailure @("forward", "--remove", "tcp:$Port") | Out-Null
Invoke-Adb @("forward", "tcp:$Port", "tcp:$Port") | Out-Null

$ClassPath = "$RemoteRoot/lifecycle-probe.jar`:$RemoteRoot/base.apk"
$LaunchCommand = "cd '$RemoteRoot' && exec env CLASSPATH='$ClassPath' LD_LIBRARY_PATH='$RemoteRoot' app_process /system/bin royale.nativehost.JniHost '$RemoteRoot' serve-direct '$Port'"
$Launch = "nohup sh -c `"$LaunchCommand`" >'$RemoteRoot/service.log' 2>&1 </dev/null &"
Invoke-Adb @("shell", $Launch) | Out-Null

$Deadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
$LastError = ""
while ([DateTime]::UtcNow -lt $Deadline) {
    try {
        $Response = Invoke-JsonRequest $Port @{op = "ping"}
        if ($Response.ok) {
            $Status = Invoke-JsonRequest $Port @{op = "status"}
            $GuestPids = @(Get-ServicePids)
            [pscustomobject]@{
                ready = $true
                mode = "serve-direct"
                slot = $Slot
                port = $Port
                serial = $Serial
                remote_root = $RemoteRoot
                guest_pids = $GuestPids
                state = $Status.state
            } | ConvertTo-Json -Depth 12
            exit 0
        }
    } catch {
        $LastError = $_.Exception.Message
    }
    Start-Sleep -Milliseconds 250
}
$Tail = Invoke-Adb -AllowFailure @("shell", "tail -n 120 '$RemoteRoot/service.log'")
throw "Direct service did not become ready: $LastError`n$($Tail | Out-String)"
