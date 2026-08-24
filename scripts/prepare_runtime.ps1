param(
    [string]$X86_64Apk = "",
    [string]$AssetPackApk = ""
)

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

$ApksDir = Require-Env "CR_SANDBOX_APKS"
$RuntimeDir = Require-Env "CR_SANDBOX_RUNTIME_DIR"
$AssetsDir = Require-Env "CR_SANDBOX_ASSETS"

if (-not $X86_64Apk) { $X86_64Apk = Join-Path $ApksDir "split_config.x86_64.apk" }
if (-not $AssetPackApk) { $AssetPackApk = Join-Path $ApksDir "split_install_time_asset_pack.apk" }

foreach ($Path in @($X86_64Apk, $AssetPackApk)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing APK input: $Path"
    }
}

$Stage = Join-Path ([IO.Path]::GetTempPath()) ("cr-sandbox-extract-" + [guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Path $Stage -Force | Out-Null
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    New-Item -ItemType Directory -Path $AssetsDir -Force | Out-Null

    # 1. Native libraries from split_config.x86_64.apk/lib/x86_64/.
    $LibStage = Join-Path $Stage "lib"
    New-Item -ItemType Directory -Path $LibStage -Force | Out-Null
    & tar.exe -xf $X86_64Apk -C $LibStage "lib/x86_64/"
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract native libs from $X86_64Apk" }
    $LibDir = Join-Path $LibStage "lib\x86_64"
    $LibFiles = @(Get-ChildItem -LiteralPath $LibDir -Filter "*.so" -ErrorAction SilentlyContinue)
    if ($LibFiles.Count -lt 14) {
        throw "Expected 14 native libraries, found $($LibFiles.Count) under $X86_64Apk/lib/x86_64/"
    }
    foreach ($File in $LibFiles) {
        Copy-Item -LiteralPath $File.FullName -Destination (Join-Path $RuntimeDir $File.Name) -Force
    }

    # 2. csv_client / csv_logic tables from the install-time asset pack.
    $AssetStage = Join-Path $Stage "assets"
    New-Item -ItemType Directory -Path $AssetStage -Force | Out-Null
    & tar.exe -xf $AssetPackApk -C $AssetStage "assets/csv_client/" "assets/csv_logic/"
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract csv tables from $AssetPackApk" }
    foreach ($Dir in @("csv_client", "csv_logic")) {
        $Source = Join-Path $AssetStage ("assets\" + $Dir)
        if (Test-Path -LiteralPath $Source -PathType Container) {
            $Destination = Join-Path $AssetsDir $Dir
            New-Item -ItemType Directory -Path $Destination -Force | Out-Null
            Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
        }
    }

    # 3. The two arena/tilemap CSVs the native resource chain requests.
    & tar.exe -xf $AssetPackApk -C $AssetStage `
        "assets/locations/training_arena.csv" "assets/tilemaps/tilemap.csv"
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract arena/tilemap CSVs from $AssetPackApk" }
    New-Item -ItemType Directory -Path (Join-Path $AssetsDir "locations") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $AssetsDir "tilemaps") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $AssetStage "assets\locations\training_arena.csv") `
        -Destination (Join-Path $AssetsDir "locations\training_arena.csv") -Force
    Copy-Item -LiteralPath (Join-Path $AssetStage "assets\tilemaps\tilemap.csv") `
        -Destination (Join-Path $AssetsDir "tilemaps\tilemap.csv") -Force

    $CsvCount = @(Get-ChildItem -LiteralPath $AssetsDir -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue).Count
    [pscustomobject]@{
        schema_version = 1
        native_lib_count = $LibFiles.Count
        runtime_dir = (Resolve-Path -LiteralPath $RuntimeDir).Path
        assets_dir = (Resolve-Path -LiteralPath $AssetsDir).Path
        csv_file_count = $CsvCount
    } | ConvertTo-Json
}
finally {
    if (Test-Path -LiteralPath $Stage) {
        Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
    }
}
