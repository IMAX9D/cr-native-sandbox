param(
    [string]$Adb = $(if ($env:CR_SANDBOX_ADB) { $env:CR_SANDBOX_ADB } else { throw "Missing CR_SANDBOX_ADB; dot-source runtime.env.ps1 first" }),
    [string]$Serial = "emulator-5554",
    [int]$Port = 37031,
    [int]$Slot = 0
)

$ErrorActionPreference = "Stop"
$RemoteRoot = "/data/local/tmp/cr-native-direct-$Slot"
$ProcessLines = & $Adb -s $Serial shell "ps -A -o PID,ARGS"
$Pids = @($ProcessLines | ForEach-Object {
    $Line = [string]$_
    if ($Line.Contains("royale.nativehost.JniHost") -and
        $Line.Contains($RemoteRoot) -and
        $Line.Contains("serve-direct")) {
        $Fields = $Line.Trim() -split "\s+", 2
        if ($Fields[0] -match '^\d+$') { [int]$Fields[0] }
    }
})
foreach ($PidValue in $Pids) {
    & $Adb -s $Serial shell "kill '$PidValue' 2>/dev/null || true"
}
& $Adb -s $Serial forward --remove "tcp:$Port" 2>$null
[pscustomobject]@{stopped = $true; slot = $Slot; port = $Port; pids = $Pids} |
    ConvertTo-Json
