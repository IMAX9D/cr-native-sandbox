package royale.nativehost;

import android.app.Activity;
import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.os.Looper;

import java.io.File;

/** Activity-shaped context proxy with no window or lifecycle side effects. */
public final class HeadlessActivity extends Activity {
    private final Context delegate;

    public HeadlessActivity(Context delegate) {
        this.delegate = delegate;
    }

    @Override
    public Context getApplicationContext() {
        return delegate.getApplicationContext();
    }

    @Override
    public ApplicationInfo getApplicationInfo() {
        return delegate.getApplicationInfo();
    }

    @Override
    public AssetManager getAssets() {
        return delegate.getAssets();
    }

    @Override
    public ClassLoader getClassLoader() {
        return delegate.getClassLoader();
    }

    @Override
    public File getCacheDir() {
        return delegate.getCacheDir();
    }

    @Override
    public File getExternalFilesDir(String type) {
        return delegate.getExternalFilesDir(type);
    }

    @Override
    public Looper getMainLooper() {
        return delegate.getMainLooper();
    }

    @Override
    public String getPackageName() {
        return delegate.getPackageName();
    }

    @Override
    public PackageManager getPackageManager() {
        return delegate.getPackageManager();
    }

    @Override
    public Resources getResources() {
        return delegate.getResources();
    }

    @Override
    public Object getSystemService(String name) {
        return delegate.getSystemService(name);
    }

}
