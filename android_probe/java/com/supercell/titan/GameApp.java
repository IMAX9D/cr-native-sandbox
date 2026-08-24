package com.supercell.titan;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.os.Looper;
import android.view.Surface;
import android.view.Display;
import android.hardware.display.DisplayManager;

import java.io.File;
import java.lang.reflect.Field;
import royale.nativehost.HeadlessApplication;

/**
 * Headless JNI contract for libg.
 *
 * <p>This class intentionally contains no UI, billing, advertising, network,
 * integrity, or lifecycle implementation. Its native declarations mirror the
 * exact runtime-150535029 GameApp descriptors so JNI_OnLoad can register the
 * original functions without initializing the protected application shell.</p>
 */
public final class GameApp extends Activity {
    private static GameApp instance;
    private final Context delegate;
    private final Intent headlessIntent = new Intent();

    public GameApp(Context delegate) {
        this.delegate = delegate;
        instance = this;
        try {
            Field application = Activity.class.getDeclaredField("mApplication");
            application.setAccessible(true);
            application.set(this, new HeadlessApplication(delegate));
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException("cannot bind headless Application", error);
        }
    }

    public static final class NotificationData {}

    public static native boolean backButtonPressed();
    public static native String createGameMain(
        AssetManager assets,
        String dataDir,
        String cacheDir,
        String externalCacheDir,
        long availableBytes,
        int width,
        int height,
        int densityDpi,
        float xdpi,
        float ydpi,
        int graphicsApi,
        String externalFilesDir,
        Activity activity
    );
    public static native void deinit();
    public static native void dialogDismissed(int first, int second);
    public static native int getAllowedScreenRotations();
    public static native void handleDeeplinkURL(String value);
    public static native void handleObservation(int value);
    public static native boolean isScreenResizeSupported();
    public static native void logDebuggerException(String value);
    public static native void nOnActivityResult(int request, int result, Intent data);
    public static native void nOnAppStart();
    public static native void nOnAppStop();
    public static native void nOnApplicationCreate();
    public static native void nOnConfigurationChanged(Configuration configuration);
    public static native void nOnCreate();
    public static native void nOnDestroy();
    public static native void nOnDisplayAdded(int display);
    public static native void nOnDisplayChanged(int display);
    public static native void nOnDisplayRemoved(int display);
    public static native void nOnKeyDownEvent(int key, int unicode);
    public static native void nOnKeyRemap(long value, int[] from, int[] to, String name);
    public static native void nOnKeyUpEvent(int key, int unicode);
    public static native void nOnMouseEvent(float x, float y);
    public static native void nOnMouseHoverEvent(float x, float y);
    public static native void nOnPause();
    public static native void nOnRestart();
    public static native void nOnResume();
    public static native void nOnStart();
    public static native void nOnStop();
    public static native void nOnSurfaceChanged(Surface surface, int width, int height);
    public static native void nOnSurfaceCreated(Surface surface);
    public static native void nOnSurfaceDestroyed(Surface surface);
    public static native void nOnTouchEvent(int action, int pointer, int x, int y);
    public static native void nOnTrimMemory(int level);
    public static native void nOnWindowFocusChanged(boolean focused);
    public static native void nRunFromUiThread(long pointer);
    private static native void notificationCanceled(NotificationData data);
    private static native void notificationDelivered(NotificationData data);
    private static native void notificationReceived(NotificationData data);
    private static native void notificationReceivedOnly(NotificationData data);
    private static native void notificationScheduled(NotificationData data);
    public static native void setDeviceVerificationResult(
        boolean first, boolean second, String value
    );
    public static native void setPushNotificationValues(
        int type, String first, String second
    );
    public static native void setSafeMargins(int left, int top, int right, int bottom);

    // Safe callback defaults used by native platform glue before a battle
    // object exists. They deliberately expose no application/account state.
    public static GameApp getInstance() {
        return instance;
    }

    public static String getAPKPath() {
        return "";
    }

    public static boolean isEmulator() {
        return false;
    }

    public static Display.Mode[] getDisplayModes() {
        Display display = getDefaultDisplay();
        return display == null ? new Display.Mode[0] : display.getSupportedModes();
    }

    public static Display.Mode getCurrentDisplayMode() {
        Display display = getDefaultDisplay();
        return display == null ? null : display.getMode();
    }

    public static void setDisplayMode(int modeId) {}

    public static boolean isPlayingUserMusic() { return false; }

    /** Headless mode never schedules Android notifications. */
    public static void cancelAllNotifications() {}

    private static Display getDefaultDisplay() {
        if (instance == null) {
            return null;
        }
        DisplayManager manager = (DisplayManager) instance.getSystemService(
            Context.DISPLAY_SERVICE
        );
        return manager == null ? null : manager.getDisplay(Display.DEFAULT_DISPLAY);
    }

    public void beforeLogicCallback() {}

    @Override
    public Intent getIntent() {
        return headlessIntent;
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
        return GameApp.class.getClassLoader();
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
