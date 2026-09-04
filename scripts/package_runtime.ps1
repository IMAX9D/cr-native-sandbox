param(
    [Parameter(Mandatory = $true)]
    [string]$ApksDirectory,

    [Parameter(Mandatory = $true)]
    [string]$RuntimeDirectory,

    [Parameter(Mandatory = $true)]
    [string]$AssetsDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$ArchiveName = "cr-native-sandbox-runtime-150535029.zip"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ManifestPath = Join-Path $ProjectRoot "bindings\runtime-manifest.json"

function Resolve-ExistingDirectory {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label directory not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-ManifestFile {
    param(
        [string]$Root,
        [object]$Entry,
        [string]$Label
    )
    $Path = Join-Path $Root ([string]$Entry.name)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Label file: $Path"
    }
    $File = Get-Item -LiteralPath $Path
    $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([long]$File.Length -ne [long]$Entry.size) {
        throw "$Label size mismatch for $($Entry.name): got $($File.Length), expected $($Entry.size)"
    }
    if ($Hash -ne [string]$Entry.sha256) {
        throw "$Label SHA-256 mismatch for $($Entry.name): got $Hash, expected $($Entry.sha256)"
    }
    return $Path
}

if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Runtime manifest not found: $ManifestPath"
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if (
    [string]$Manifest.runtime_version -ne "150535029" -or
    [string]$Manifest.game_version -ne "15.535.29" -or
    [string]$Manifest.abi -ne "x86_64"
) {
    throw "The checked-in manifest is not the supported 150535029 x86_64 runtime"
}

$ApksDirectory = Resolve-ExistingDirectory $ApksDirectory "APK"
$RuntimeDirectory = Resolve-ExistingDirectory $RuntimeDirectory "native library"
$AssetsDirectory = Resolve-ExistingDirectory $AssetsDirectory "asset"
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

$ArchivePath = Join-Path $OutputDirectory $ArchiveName
$StageRoot = Join-Path $OutputDirectory (
    [IO.Path]::GetFileNameWithoutExtension($ArchiveName) + ".expanded"
)
$RuntimeRoot = Join-Path $StageRoot "runtime"
if (Test-Path -LiteralPath $ArchivePath) {
    throw "Refusing to overwrite existing archive: $ArchivePath"
}
if (Test-Path -LiteralPath $StageRoot) {
    throw "Refusing to overwrite existing staging directory: $StageRoot"
}

$ApkDestination = Join-Path $RuntimeRoot "apks"
$LibDestination = Join-Path $RuntimeRoot "x86_64-libs"
$AssetDestination = Join-Path $RuntimeRoot "extracted-assets"
New-Item -ItemType Directory -Path $ApkDestination -Force | Out-Null
New-Item -ItemType Directory -Path $LibDestination -Force | Out-Null
New-Item -ItemType Directory -Path $AssetDestination -Force | Out-Null

foreach ($Entry in $Manifest.apks) {
    $Source = Assert-ManifestFile $ApksDirectory $Entry "APK"
    Copy-Item -LiteralPath $Source -Destination (Join-Path $ApkDestination $Entry.name)
}
foreach ($Entry in $Manifest.native_libs) {
    $Source = Assert-ManifestFile $RuntimeDirectory $Entry "native library"
    Copy-Item -LiteralPath $Source -Destination (Join-Path $LibDestination $Entry.name)
}

foreach ($Name in @("csv_client", "csv_logic")) {
    $Source = Join-Path $AssetsDirectory $Name
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Missing DataTable directory: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $AssetDestination -Recurse
}

$TableCount = @(
    Get-ChildItem -LiteralPath (Join-Path $AssetDestination "csv_client") -File -Recurse
    Get-ChildItem -LiteralPath (Join-Path $AssetDestination "csv_logic") -File -Recurse
).Count
if ($TableCount -ne [int]$Manifest.assets.expected_table_files) {
    throw "DataTable file count mismatch: got $TableCount, expected $($Manifest.assets.expected_table_files)"
}

$MapSources = @(
    @{
        Source = Join-Path $AssetsDirectory "locations\training_arena.csv"
        Destination = Join-Path $AssetDestination "locations\training_arena.csv"
        ArchiveEntry = "assets/locations/training_arena.csv"
    },
    @{
        Source = Join-Path $AssetsDirectory "tilemaps\tilemap.csv"
        Destination = Join-Path $AssetDestination "tilemaps\tilemap.csv"
        ArchiveEntry = "assets/tilemaps/tilemap.csv"
    }
)
$MissingMaps = @($MapSources | Where-Object {
    -not (Test-Path -LiteralPath $_.Source -PathType Leaf)
})
if ($MissingMaps.Count -gt 0) {
    $AssetPack = Join-Path $ApksDirectory "split_install_time_asset_pack.apk"
    foreach ($Map in $MissingMaps) {
        $DestinationDirectory = Split-Path -Parent $Map.Destination
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
        & tar.exe -xf $AssetPack -C $DestinationDirectory --strip-components 2 $Map.ArchiveEntry
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Map.Destination -PathType Leaf)) {
            throw "Failed to extract $($Map.ArchiveEntry) from the asset pack"
        }
    }
}
foreach ($Map in $MapSources) {
    if (Test-Path -LiteralPath $Map.Source -PathType Leaf) {
        $DestinationDirectory = Split-Path -Parent $Map.Destination
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
        Copy-Item -LiteralPath $Map.Source -Destination $Map.Destination
    }
    if (-not (Test-Path -LiteralPath $Map.Destination -PathType Leaf)) {
        throw "Missing required map file: $($Map.Destination)"
    }
}

$Readme = @"
CR Native Core Runtime
======================

Runtime version: 150535029
Game version:    15.535.29
ABI:             x86_64
libg SHA-256:    $($Manifest.frozen_libg_sha256)

This archive contains the version-pinned runtime inputs expected by:
https://github.com/IMAX9D/cr-native-sandbox

Usage:
1. Clone the GitHub repository.
2. Extract this ZIP directly into the repository root.
3. Confirm that <repository>\runtime\apks and x86_64-libs exist.
4. Copy runtime.env.example.ps1 to runtime.env.ps1 and edit toolchain paths.
5. Dot-source runtime.env.ps1.
6. Run scripts\doctor.ps1.
7. Run scripts\smoke.ps1 only when you intentionally want to start the headless AVD.

Do not combine these files with another game version or ABI. The checked-in
host uses version-specific RVAs and must fail closed on a mismatch.

SHA256SUMS.txt covers every file in runtime\ except SHA256SUMS.txt itself.
"@
$ReadmePath = Join-Path $RuntimeRoot "README.txt"
[IO.File]::WriteAllText(
    $ReadmePath,
    $Readme.Trim() + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

$ChecksumPath = Join-Path $RuntimeRoot "SHA256SUMS.txt"
$Rows = Get-ChildItem -LiteralPath $RuntimeRoot -File -Recurse |
    Where-Object { $_.FullName -ne $ChecksumPath } |
    Sort-Object FullName |
    ForEach-Object {
        $Relative = $_.FullName.Substring($RuntimeRoot.Length + 1).Replace("\", "/")
        $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$Hash  $Relative"
    }
[IO.File]::WriteAllText(
    $ChecksumPath,
    ($Rows -join [Environment]::NewLine) + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory(
    $StageRoot,
    $ArchivePath,
    [IO.Compression.CompressionLevel]::Optimal,
    $false
)

$Archive = Get-Item -LiteralPath $ArchivePath
$ArchiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$Sidecar = [ordered]@{
    schema_version = 1
    kind = "cr_native_runtime_bundle_receipt_v1"
    archive = $Archive.Name
    archive_size = [long]$Archive.Length
    archive_sha256 = $ArchiveHash
    game_version = [string]$Manifest.game_version
    runtime_version = [string]$Manifest.runtime_version
    abi = [string]$Manifest.abi
    libg_sha256 = [string]$Manifest.frozen_libg_sha256
    apk_count = @($Manifest.apks).Count
    native_lib_count = @($Manifest.native_libs).Count
    table_file_count = $TableCount
    map_file_count = 2
    packaged_file_count = @(
        Get-ChildItem -LiteralPath $RuntimeRoot -File -Recurse
    ).Count
    archive_layout = "runtime/"
    source_manifest = $ManifestPath
}
$SidecarPath = $ArchivePath + ".manifest.json"
[IO.File]::WriteAllText(
    $SidecarPath,
    ($Sidecar | ConvertTo-Json -Depth 6) + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)
$Sidecar | ConvertTo-Json -Depth 6
