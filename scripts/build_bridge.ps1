param(
    [string]$NdkRoot = "D:\Codex\toolchains\android-ndk-r27d",
    [int]$ApiLevel = 23
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Compiler = Join-Path $NdkRoot "toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe"
$JniHeaders = Join-Path $NdkRoot "toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\include"
$Source = Join-Path $ProjectRoot "android_probe\native\jni_bridge.cpp"
$ArtifactRoot = Join-Path $ProjectRoot "artifacts"
$Output = Join-Path $ArtifactRoot "libnative_core_probe.so"
foreach ($Path in @($Compiler, $Source)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing bridge build dependency: $Path"
    }
}
New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
& $Compiler `
    "--target=x86_64-linux-android$ApiLevel" `
    -std=c++20 `
    -fPIC `
    -shared `
    -O2 `
    -g `
    -Wall `
    -Wextra `
    -Werror `
    "-I$JniHeaders" `
    $Source `
    -ldl `
    -o $Output
if ($LASTEXITCODE -ne 0) { throw "bridge build failed: $LASTEXITCODE" }
[pscustomobject]@{
    output = $Output
    target = "x86_64-linux-android$ApiLevel"
    sha256 = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json
