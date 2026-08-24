package com.supercell.titan;

import android.content.Context;
import android.content.Intent;

/** Fixed charged-battery bridge with no BroadcastReceiver. */
public final class BatteryMonitor {
    public static final class Snapshot {
        public final int level;
        public final boolean charging;

        public Snapshot(int level, boolean charging) {
            this.level = level;
            this.charging = charging;
        }
    }

    public final Context context;
    public final long nativeHandle;
    public boolean running;

    public BatteryMonitor(Context context, long nativeHandle) {
        this.context = context;
        this.nativeHandle = nativeHandle;
    }

    public static Snapshot fromIntent(Intent intent) { return new Snapshot(100, true); }
    private static native void updateBatteryInfo(
        long nativeHandle, int level, boolean charging
    );
    public Snapshot getOnce() { return new Snapshot(100, true); }
    public final void notifyNative(Snapshot snapshot) {
        updateBatteryInfo(nativeHandle, snapshot.level, snapshot.charging);
    }
    public void start() {
        running = true;
        notifyNative(getOnce());
    }
    public void stop() { running = false; }
}
