# Copy to runtime.env.ps1 and edit the local paths. Never commit runtime.env.ps1.
#
# Dot-source it in every shell before running the scripts:
#     . .\runtime.env.ps1
#
# Verified toolchain baseline (frozen compatibility matrix):
#   Android Emulator 37.1.11
#   Platform-Tools / ADB 37.0.1
#   system-images;android-31;default;x86_64  rev 3
#   platforms;android-35  rev 2,  build-tools 35.0.0
#   NDK 27.3.13750724 (r27d)
#   JDK 17.0.20.1
#   Python >= 3.11

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- Android SDK / toolchain ----------------------------------------------
$env:CR_SANDBOX_ANDROID_SDK   = "C:\Android\Sdk"
$env:CR_SANDBOX_ADB           = "$env:CR_SANDBOX_ANDROID_SDK\platform-tools\adb.exe"
$env:CR_SANDBOX_ANDROID_TOOLS = "$env:CR_SANDBOX_ANDROID_SDK\cmdline-tools\latest"
$env:CR_SANDBOX_ANDROID_JAR   = "$env:CR_SANDBOX_ANDROID_SDK\platforms\android-35\android.jar"
$env:CR_SANDBOX_NDK           = "$env:CR_SANDBOX_ANDROID_SDK\ndk\27.3.13750724"
$env:CR_SANDBOX_JDK           = "C:\Program Files\Eclipse Adoptium\jdk-17"

# --- AVD ------------------------------------------------------------------
# Must use the AOSP default image (rootable); Play Store images are rejected.
$env:CR_SANDBOX_AVD_HOME      = "$env:LOCALAPPDATA\Android\avd"
$env:CR_SANDBOX_AVD_NAME      = "royale_worker_api31"
$env:CR_SANDBOX_SYSTEM_IMAGE  = "system-images;android-31;default;x86_64"

# --- Legally obtained runtime (15.535.29 / 150535029) ---------------------
# The default layout lives under the git-ignored runtime/ directory. Put the
# five split APKs in runtime\apks\, then let scripts\prepare_runtime.ps1 and
# scripts\freeze_runtime.ps1 populate the rest.
$env:CR_SANDBOX_APKS           = "$Root\runtime\apks"
$env:CR_SANDBOX_RUNTIME_DIR    = "$Root\runtime\x86_64-libs"
$env:CR_SANDBOX_BASE_APK       = "$env:CR_SANDBOX_APKS\base.apk"
$env:CR_SANDBOX_ASSET_PACK_APK = "$env:CR_SANDBOX_APKS\split_install_time_asset_pack.apk"
$env:CR_SANDBOX_ASSETS         = "$Root\runtime\extracted-assets"

# --- Writable outputs (kept outside the repository) -----------------------
$env:CR_SANDBOX_DATA           = "$env:LOCALAPPDATA\cr-native-sandbox\data"
