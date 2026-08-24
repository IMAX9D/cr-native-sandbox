# Copy to runtime.env.ps1 and edit local paths. Never commit runtime.env.ps1.

$env:CR_SANDBOX_ANDROID_SDK = "C:\Android\Sdk"
$env:CR_SANDBOX_ADB = "$env:CR_SANDBOX_ANDROID_SDK\platform-tools\adb.exe"
$env:CR_SANDBOX_ANDROID_TOOLS = "$env:CR_SANDBOX_ANDROID_SDK\cmdline-tools\latest"
$env:CR_SANDBOX_ANDROID_JAR = "$env:CR_SANDBOX_ANDROID_SDK\platforms\android-35\android.jar"
$env:CR_SANDBOX_NDK = "C:\Android\Sdk\ndk\27.3.13750724"
$env:CR_SANDBOX_JDK = "C:\Program Files\Eclipse Adoptium\jdk-17"
$env:CR_SANDBOX_AVD_HOME = "D:\Android\avd"

# Legally obtained, exact 15.535.29 / 150535029 runtime inputs.
$env:CR_SANDBOX_APKS = "D:\CR-runtime\apks"
$env:CR_SANDBOX_RUNTIME_DIR = "D:\CR-runtime\x86_64-libs"
$env:CR_SANDBOX_BASE_APK = "$env:CR_SANDBOX_APKS\base.apk"
$env:CR_SANDBOX_ASSET_PACK_APK = "$env:CR_SANDBOX_APKS\split_install_time_asset_pack.apk"
$env:CR_SANDBOX_ASSETS = "D:\CR-runtime\extracted-assets"

# Writable local outputs. Do not place this inside the Git repository.
$env:CR_SANDBOX_DATA = "D:\CR-sandbox-data"
