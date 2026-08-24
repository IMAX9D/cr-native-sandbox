param(
    [string]$ManifestTemplate = "",
    [string]$OutputManifest = ""
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
$DataDir = Require-Env "CR_SANDBOX_DATA"

if (-not $ManifestTemplate) {
    $ManifestTemplate = Join-Path $ProjectRoot "bindings\runtime-manifest.json"
}
if (-not $OutputManifest) {
    $OutputManifest = Join-Path $DataDir "manifest\runtime-manifest.json"
}

if (-not (Test-Path -LiteralPath $ManifestTemplate -PathType Leaf)) {
    throw "Runtime manifest template not found: $ManifestTemplate"
}
$Manifest = Get-Content -LiteralPath $ManifestTemplate -Raw | ConvertFrom-Json
$FrozenLibg = [string]$Manifest.frozen_libg_sha256

# --- APKs ---------------------------------------------------------------
foreach ($Apk in $Manifest.apks) {
    $Path = Join-Path $ApksDir $Apk.name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing APK: $Path"
    }
    $File = Get-Item -LiteralPath $Path
    $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Apk.size -and [long]$Apk.size -ne [long]$File.Length) {
        throw "$($Apk.name) size mismatch: got $($File.Length), expected $($Apk.size)"
    }
    if ($Apk.sha256 -and [string]$Apk.sha256 -ne $Hash) {
        throw "$($Apk.name) SHA-256 mismatch: got $Hash, expected $($Apk.sha256)"
    }
    $Apk.size = [long]$File.Length
    $Apk.sha256 = $Hash
}

# --- Native libraries ---------------------------------------------------
foreach ($Lib in $Manifest.native_libs) {
    $Path = Join-Path $RuntimeDir $Lib.name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing native library: $Path"
    }
    $File = Get-Item -LiteralPath $Path
    $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Lib.size -and [long]$Lib.size -ne [long]$File.Length) {
        throw "$($Lib.name) size mismatch: got $($File.Length), expected $($Lib.size)"
    }
    if ($Lib.sha256 -and [string]$Lib.sha256 -ne $Hash) {
        throw "$($Lib.name) SHA-256 mismatch: got $Hash, expected $($Lib.sha256)"
    }
    $Lib.size = [long]$File.Length
    $Lib.sha256 = $Hash
    if ($Lib.name -eq "libg.so" -and $Hash -ne $FrozenLibg) {
        throw "libg.so hash mismatch: got $Hash, expected $FrozenLibg. The runtime does not match the frozen build and the sandbox must fail closed."
    }
}

New-Item -ItemType Directory -Path (Split-Path -Parent $OutputManifest) -Force | Out-Null
$Manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $OutputManifest -Encoding utf8

[pscustomobject]@{
    schema_version = 1
    frozen = $true
    output = (Resolve-Path -LiteralPath $OutputManifest).Path
    libg_sha256 = $FrozenLibg
    apk_count = @($Manifest.apks).Count
    native_lib_count = @($Manifest.native_libs).Count
} | ConvertTo-Json
