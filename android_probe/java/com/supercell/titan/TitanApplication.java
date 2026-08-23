package com.supercell.titan;

import android.content.Context;

/** Minimal application-side contract required before GameApp starts Mainloop. */
public final class TitanApplication {
    private static Context appContext;

    private TitanApplication() {}

    public static native void nOnNativeLibrariesLoaded();

    public static void bindContext(Context context) {
        appContext = context;
    }

    public static Context getAppContext() {
        return appContext;
    }

    public static boolean isInForeground() {
        return true;
    }
}
