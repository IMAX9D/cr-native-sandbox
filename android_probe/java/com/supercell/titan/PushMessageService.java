package com.supercell.titan;

/** Offline push-notification bridge with no permission or FCM access. */
public final class PushMessageService {
    public static final int $r8$clinit = 0;

    public static void getToken(boolean forceRefresh) {}
    public static boolean hasPermissionBeenRequested() { return true; }
    public static boolean isPermissionGranted() { return false; }
    public static void onDestroy(GameApp app) {}
    public static void register() {}
    public static void requestPermission(RequestPermissionListener listener) {
        if (listener != null) { listener.onPermissionResult(false); }
    }
    public static void requestToken() {}
    public static boolean shouldShowPermissionRationale() { return false; }
}
