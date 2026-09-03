[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Compiler = 'D:\Codex\toolchains\android-ndk-r27d\toolchains\llvm\prebuilt\windows-x86_64\bin\x86_64-linux-android24-clang.cmd'
$Source = Join-Path $Root 'native_core\mumu_arm64_chain_scan.c'
$OutputDirectory = Join-Path $Root 'artifacts\mumu-live'
$Output = Join-Path $OutputDirectory 'mumu-arm64-chain-scan-x86_64'
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
foreach ($Path in @($Compiler, $Source)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ARM64 chain scan build dependency: $Path"
    }
}
& $Compiler '-O3' '-std=c17' '-fPIE' '-pie' '-s' '-o' $Output $Source
if ($LASTEXITCODE -ne 0) { throw "ARM64 chain scan build failed: $LASTEXITCODE" }
$item = Get-Item -LiteralPath $Output
$hash = Get-FileHash -LiteralPath $Output -Algorithm SHA256
[ordered]@{output=$item.FullName;bytes=$item.Length;sha256=$hash.Hash.ToLowerInvariant()} | ConvertTo-Json
