param(
    [string]$RunsRoot = "D:\AI_data\cr-native-core\training\runs",
    [string]$AuditRoot = "D:\AI_data\cr-native-core\maintenance",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

# This is deliberately an exact allow-list.  A future run never inherits a broad
# wildcard that could catch a formal Self-Play run.
$DeleteNames = @(
    "smoke-20260823T062921Z",
    "debug-smoke-1",
    "smoke-20260823T063001Z",
    "smoke-20260823T063036Z",
    "smoke-20260823T063225Z",
    "smoke-20260823T063619Z-09737220",
    "smoke-20260823T063913Z-1f006968",
    "throughput-baseline-profile",
    "bench-baseline-fixed",
    "bench-persistent-fixed",
    "bench-persistent-nodelay-fixed",
    "bench-compact-fixed",
    "bench-mask-cache-fixed",
    "bench-vector-fixed",
    "bench-vector-sync-fixed",
    "bench-skip-episode-scan-fixed",
    "bench-skip-episode-scan-fixed-v2",
    "bench-vector-confirm-fixed",
    "bench-vector-cpu-fixed",
    "smoke-20260823T083904Z-f1d7f42f",
    "scale-2w-a",
    "scale-2w-b",
    "scale-2w-c",
    "scale-2w-d",
    "scale-4w-a",
    "scale-4w-b",
    "scale-2w-e",
    "scale-4w-c",
    "scale-2w-f",
    "scale-4w-resource",
    "jni-profile-2w",
    "transport-redir-2w",
    "transport-redir-2w-b",
    "transport-adb-control",
    "transport-redir-2w-c",
    "transport-adb-control-b",
    "transport-redir-4w",
    "transport-adb-4w-control",
    "optimized-direct-4w",
    "optimized-direct-4w-b",
    "cuda-graph-4w-a",
    "cuda-eager-4w-control",
    "cuda-graph-4w-b",
    "cuda-graph-2w",
    "avd-8cpu-4w-a",
    "avd-8cpu-4w-b",
    "vector-sample-4w-a",
    "hybrid-sample-4w-a",
    "hybrid-sample-4w-b",
    "final-fixed-4w",
    "smoke-direct-graph",
    "latency-multiwave-check",
    "final-direct-graph-smoke"
)

$resolvedRoot = [System.IO.Path]::GetFullPath($RunsRoot).TrimEnd('\')
$requiredRoot = [System.IO.Path]::GetFullPath(
    "D:\AI_data\cr-native-core\training\runs"
).TrimEnd('\')
if ($resolvedRoot -ne $requiredRoot) {
    throw "Refusing cleanup outside the audited runs root: $resolvedRoot"
}
if (-not (Test-Path -LiteralPath $resolvedRoot -PathType Container)) {
    throw "Runs root does not exist: $resolvedRoot"
}

$entries = foreach ($name in $DeleteNames) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot $name))
    if ([System.IO.Path]::GetDirectoryName($target) -ne $resolvedRoot) {
        throw "Unsafe cleanup target: $target"
    }
    if (-not (Test-Path -LiteralPath $target -PathType Container)) {
        continue
    }
    $files = Get-ChildItem -LiteralPath $target -Recurse -File -Force
    [pscustomobject]@{
        name = $name
        path = $target
        bytes = [int64](($files | Measure-Object Length -Sum).Sum)
        file_count = [int](($files | Measure-Object).Count)
    }
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$mode = if ($Execute) { "executed" } else { "dry-run" }
$audit = [ordered]@{
    schema_version = 1
    kind = "training_scratch_cleanup"
    utc = (Get-Date).ToUniversalTime().ToString("o")
    mode = $mode
    runs_root = $resolvedRoot
    entry_count = @($entries).Count
    total_bytes = [int64](($entries | Measure-Object bytes -Sum).Sum)
    entries = @($entries)
}

New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
$auditPath = Join-Path $AuditRoot "cleanup-$stamp-$mode.json"
$audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding utf8

if ($Execute) {
    foreach ($entry in $entries) {
        $verified = [System.IO.Path]::GetFullPath($entry.path)
        if ([System.IO.Path]::GetDirectoryName($verified) -ne $resolvedRoot) {
            throw "Cleanup target failed final parent check: $verified"
        }
        Remove-Item -LiteralPath $verified -Recurse -Force
    }
}

[pscustomobject]@{
    mode = $mode
    deleted_directories = if ($Execute) { @($entries).Count } else { 0 }
    selected_directories = @($entries).Count
    selected_bytes = [int64]$audit.total_bytes
    audit = $auditPath
} | ConvertTo-Json -Compress
