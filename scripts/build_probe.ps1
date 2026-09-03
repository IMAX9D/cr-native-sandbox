param(
    [string]$JdkRoot = $(if ($env:CR_SANDBOX_JDK) { $env:CR_SANDBOX_JDK } else { throw "Missing CR_SANDBOX_JDK; copy and dot-source runtime.env.ps1 first" }),
    [string]$AndroidCommandLineTools = $(if ($env:CR_SANDBOX_ANDROID_TOOLS) { $env:CR_SANDBOX_ANDROID_TOOLS } else { throw "Missing CR_SANDBOX_ANDROID_TOOLS; copy and dot-source runtime.env.ps1 first" }),
    [string]$AndroidJar = $(if ($env:CR_SANDBOX_ANDROID_JAR) { $env:CR_SANDBOX_ANDROID_JAR } else { throw "Missing CR_SANDBOX_ANDROID_JAR; copy and dot-source runtime.env.ps1 first" })
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $ProjectRoot "android_probe\java"
$ArtifactRoot = Join-Path $ProjectRoot "artifacts"
$Classes = Join-Path $ArtifactRoot "probe-classes"
$Output = Join-Path $ArtifactRoot "lifecycle-probe.jar"
$Javac = Join-Path $JdkRoot "bin\javac.exe"
$Java = Join-Path $JdkRoot "bin\java.exe"
$D8Jar = Join-Path $AndroidCommandLineTools "lib\r8.jar"
$AndroidTestMockJar = Join-Path (Split-Path -Parent $AndroidJar) "optional\android.test.mock.jar"

foreach ($Path in @($Javac, $Java, $D8Jar, $AndroidJar, $AndroidTestMockJar)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing probe build dependency: $Path"
    }
}
New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
$ResolvedArtifacts = [IO.Path]::GetFullPath($ArtifactRoot)
$ResolvedClasses = [IO.Path]::GetFullPath($Classes)
if (-not $ResolvedClasses.StartsWith($ResolvedArtifacts + [IO.Path]::DirectorySeparatorChar)) {
    throw "Refusing to clean outside the artifact directory: $ResolvedClasses"
}
if (Test-Path -LiteralPath $ResolvedClasses -PathType Container) {
    Remove-Item -LiteralPath $ResolvedClasses -Recurse -Force
}
New-Item -ItemType Directory -Path $ResolvedClasses -Force | Out-Null

$Sources = @(Get-ChildItem -LiteralPath $SourceRoot -Recurse -Filter "*.java")
$CompileClasspath = "$AndroidJar;$AndroidTestMockJar"
& $Javac --release 8 -g -classpath $CompileClasspath -d $ResolvedClasses $Sources.FullName
if ($LASTEXITCODE -ne 0) { throw "javac failed: $LASTEXITCODE" }
$ClassFiles = @(Get-ChildItem -LiteralPath $ResolvedClasses -Recurse -Filter "*.class")
& $Java -cp $D8Jar com.android.tools.r8.D8 --debug --min-api 23 `
    --lib $AndroidJar --lib $AndroidTestMockJar --output $Output $ClassFiles.FullName
if ($LASTEXITCODE -ne 0) { throw "D8 failed: $LASTEXITCODE" }

[pscustomobject]@{
    output = $Output
    source_count = $Sources.Count
    sha256 = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json
