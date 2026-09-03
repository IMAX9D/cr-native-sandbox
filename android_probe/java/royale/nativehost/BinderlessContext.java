package royale.nativehost;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.os.Looper;
import android.util.DisplayMetrics;

import java.io.File;
import java.lang.reflect.Method;

/** Minimal package Context for ART processes that intentionally have no Binder driver. */
public final class BinderlessContext extends ContextWrapper {
    private final String packageName;
    private final File root;
    private final AssetManager assets;
    private final Resources resources;
    private final ApplicationInfo applicationInfo;
    private final PackageManager packageManager;

    public BinderlessContext(String runtimeRoot, String packageName) throws Exception {
        super(null);
        this.packageName = packageName;
        this.root = new File(runtimeRoot);
        java.lang.reflect.Constructor<AssetManager> constructor =
            AssetManager.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        this.assets = constructor.newInstance();
        Method addAssetPath = AssetManager.class.getDeclaredMethod(
            "addAssetPath", String.class
        );
        addAssetPath.setAccessible(true);
        int cookie = ((Integer) addAssetPath.invoke(
            assets, new File(root, "base.apk").getAbsolutePath()
        )).intValue();
        if (cookie == 0) {
            throw new IllegalStateException("cannot add base.apk to AssetManager");
        }
        File assetPack = new File(root, "asset-pack.apk");
        if (assetPack.isFile()) {
            int assetPackCookie = ((Integer) addAssetPath.invoke(
                assets, assetPack.getAbsolutePath()
            )).intValue();
            if (assetPackCookie == 0) {
                throw new IllegalStateException(
                    "cannot add asset-pack.apk to AssetManager"
                );
            }
        }
        DisplayMetrics metrics = new DisplayMetrics();
        metrics.widthPixels = 1080;
        metrics.heightPixels = 2400;
        metrics.densityDpi = 420;
        metrics.density = 420.0f / 160.0f;
        metrics.scaledDensity = metrics.density;
        metrics.xdpi = 420.0f;
        metrics.ydpi = 420.0f;
        this.resources = new Resources(assets, metrics, new Configuration());
        this.applicationInfo = new ApplicationInfo();
        applicationInfo.packageName = packageName;
        applicationInfo.sourceDir = new File(root, "base.apk").getAbsolutePath();
        applicationInfo.publicSourceDir = applicationInfo.sourceDir;
        applicationInfo.dataDir = new File(root, "data").getAbsolutePath();
        applicationInfo.nativeLibraryDir = root.getAbsolutePath();
        this.packageManager = new BinderlessPackageManager(
            packageName, applicationInfo
        );
    }

    @Override
    public Context getApplicationContext() {
        return this;
    }

    @Override
    public ApplicationInfo getApplicationInfo() {
        return applicationInfo;
    }

    @Override
    public AssetManager getAssets() {
        return assets;
    }

    @Override
    public ClassLoader getClassLoader() {
        return BinderlessContext.class.getClassLoader();
    }

    @Override
    public File getCacheDir() {
        return directory("cache");
    }

    @Override
    public File getCodeCacheDir() {
        return directory("code-cache");
    }

    @Override
    public File getDataDir() {
        return directory("data");
    }

    @Override
    public File getExternalCacheDir() {
        return directory("external-cache");
    }

    @Override
    public File getExternalFilesDir(String type) {
        return directory(type == null ? "external" : "external/" + type);
    }

    @Override
    public File getFilesDir() {
        return directory("files");
    }

    @Override
    public Looper getMainLooper() {
        return Looper.getMainLooper();
    }

    @Override
    public String getPackageName() {
        return packageName;
    }

    @Override
    public String getPackageCodePath() {
        return applicationInfo.sourceDir;
    }

    @Override
    public String getPackageResourcePath() {
        return applicationInfo.publicSourceDir;
    }

    @Override
    public PackageManager getPackageManager() {
        return packageManager;
    }

    @Override
    public Resources getResources() {
        return resources;
    }

    @Override
    public Object getSystemService(String name) {
        return null;
    }

    @Override
    public Context createPackageContext(String requestedPackage, int flags)
        throws PackageManager.NameNotFoundException {
        if (!packageName.equals(requestedPackage)) {
            throw new PackageManager.NameNotFoundException(requestedPackage);
        }
        return this;
    }

    private File directory(String relative) {
        File value = new File(root, relative);
        if (!value.exists() && !value.mkdirs()) {
            throw new IllegalStateException("cannot create " + value);
        }
        return value;
    }
}
