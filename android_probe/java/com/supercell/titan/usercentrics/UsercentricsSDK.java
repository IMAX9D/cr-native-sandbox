package com.supercell.titan.usercentrics;

import android.app.Activity;

/** Headless consent-SDK contract used by libg's Android platform layer. */
public final class UsercentricsSDK {
    public UsercentricsSDK() {}

    public void getConsentOrShowDialog(Activity activity, int buttonLayout) {
        gotConsent("[]", false, false);
    }

    public native void gotConsent(String consentJson, boolean newlyGranted, boolean anyGranted);

    public native void gotFailure();

    public native void popupClosed();

    public native void popupShowing();

    public void reset() {}

    public void showDialog(Activity activity, int buttonLayout) {
        gotConsent("[]", false, false);
    }
}
