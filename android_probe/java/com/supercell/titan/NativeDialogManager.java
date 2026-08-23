package com.supercell.titan;

import android.app.Dialog;
import android.app.DialogFragment;
import android.content.DialogInterface;
import android.os.Bundle;

import java.util.Vector;

/** Window-free dialog bridge used by the headless lifecycle. */
public class NativeDialogManager extends DialogFragment {
    public static final String VISIBLE_NATIVE_DIALOG_TAG = "headless";
    public static NativeDialogManager currentDialog;
    public static int dismissedButtonID;
    public static int dismissedDialogID;
    public static boolean doDialogDismiss;
    public static int idCounter;
    public static final Vector<Object> pendingDialogs = new Vector<>();

    public boolean dismissed;
    public int id = -1;

    public static int ShowDialog(
        String title, String message, String first, String second, String third
    ) {
        return ++idCounter;
    }

    public static void ShowPostDialog(String title, String value) {}
    public static void ShowPostURLDialog(String title, String value, String url) {}
    public static void ShowShareFilesDialog(String title, String[] paths) {}
    public static boolean isDialogVisible() { return false; }

    public static void nativeDialogDismissAll() {
        pendingDialogs.clear();
        currentDialog = null;
        doDialogDismiss = false;
    }

    public final void dialogDismissed(int dialog, int button) {}

    @Override
    public Dialog onCreateDialog(Bundle state) { return null; }

    @Override
    public void onDismiss(DialogInterface dialog) {}

    public void startupDismiss() {}
}
