[CmdletBinding()]
param(
    [string]$NdkRoot = '',
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $NdkRoot) { $NdkRoot = $env:CR_SANDBOX_NDK }
if (-not $NdkRoot) { $NdkRoot = $env:ANDROID_NDK_ROOT }
if (-not $NdkRoot -and (Test-Path -LiteralPath 'D:\Codex\toolchains\android-ndk-r27d')) {
    $NdkRoot = 'D:\Codex\toolchains\android-ndk-r27d'
}
if (-not $NdkRoot) { throw 'Set -NdkRoot, CR_SANDBOX_NDK or ANDROID_NDK_ROOT to an installed Android NDK.' }
$Compiler = Join-Path $NdkRoot 'toolchains\llvm\prebuilt\windows-x86_64\bin\x86_64-linux-android24-clang.cmd'
$Source = Join-Path $Root 'native_core\mumu_live_private_sampler.c'
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $Root 'artifacts\mumu-live' }
$Output = Join-Path $OutputDirectory 'mumu-live-reader-v2-x86_64'

foreach ($Path in @($Compiler, $Source)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing MuMu live-controller build dependency: $Path"
    }
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
& $Compiler '-O3' '-std=c17' '-Wall' '-Wextra' '-fPIE' '-pie' '-s' '-o' $Output $Source
if ($LASTEXITCODE -ne 0) {
    throw "MuMu private sampler build failed: $LASTEXITCODE"
}
$item = Get-Item -LiteralPath $Output
$hash = Get-FileHash -LiteralPath $Output -Algorithm SHA256
[ordered]@{
    output = $item.FullName
    bytes = $item.Length
    sha256 = $hash.Hash.ToLowerInvariant()
} | ConvertTo-Json
