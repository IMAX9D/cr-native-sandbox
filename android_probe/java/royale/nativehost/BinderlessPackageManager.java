package royale.nativehost;

import android.content.pm.ApplicationInfo;
import android.content.pm.FeatureInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.test.mock.MockPackageManager;

/** Deterministic package metadata used by libg's device-capability probes. */
public final class BinderlessPackageManager extends MockPackageManager {
    private final String packageName;
    private final ApplicationInfo applicationInfo;

    BinderlessPackageManager(
        String packageName, ApplicationInfo applicationInfo
    ) {
        this.packageName = packageName;
        this.applicationInfo = applicationInfo;
    }

    @Override
    public boolean hasSystemFeature(String name) {
        return false;
    }

    @Override
    public boolean hasSystemFeature(String name, int version) {
        return false;
    }

    @Override
    public FeatureInfo[] getSystemAvailableFeatures() {
        return new FeatureInfo[0];
    }

    @Override
    public ApplicationInfo getApplicationInfo(String requestedPackage, int flags)
        throws PackageManager.NameNotFoundException {
        requirePackage(requestedPackage);
        return applicationInfo;
    }

    @Override
    public PackageInfo getPackageInfo(String requestedPackage, int flags)
        throws PackageManager.NameNotFoundException {
        requirePackage(requestedPackage);
        PackageInfo value = new PackageInfo();
        value.packageName = packageName;
        value.applicationInfo = applicationInfo;
        value.versionName = "15.535.29";
        value.versionCode = 150535029;
        return value;
    }

    @Override
    public CharSequence getApplicationLabel(ApplicationInfo info) {
        return "Clash Royale";
    }

    @Override
    public String getInstallerPackageName(String requestedPackage) {
        return null;
    }

    private void requirePackage(String requestedPackage)
        throws PackageManager.NameNotFoundException {
        if (!packageName.equals(requestedPackage)) {
            throw new PackageManager.NameNotFoundException(requestedPackage);
        }
    }
}
