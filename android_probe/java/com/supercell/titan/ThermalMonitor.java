package com.supercell.titan;

import android.content.Context;

/** Fixed no-throttling thermal bridge for deterministic headless execution. */
public final class ThermalMonitor {
    public static final int THERMAL_PREDICTION_INITIALIZED = 0;
    public static final int THERMAL_PREDICTION_NOT_AVAILABLE = 1;
    public static final int THERMAL_PREDICTION_NOT_INITIALIZED = 2;
    public static final int THERMAL_PREDICTION_ONGOING = 3;
    public static ThermalMonitor instance = new ThermalMonitor();

    public static final class SessionResult {
        public int maxStatus;
        public int minStatus;
        public int statusAtEnd;
        public int statusAtStart;
        public boolean success;
    }

    public static int getCurrentThermalStatus(Context context) { return 0; }
    public static synchronized ThermalMonitor getInstance() { return instance; }
    public static boolean isThermalStatusAvailable() { return false; }
    public static String thermalStatusToString(int status) { return "none"; }

    public void deinitThermalPrediction() {}
    public SessionResult endSession(Context context, String name) {
        return new SessionResult();
    }
    public int getThermalPredictionStatus() {
        return THERMAL_PREDICTION_NOT_AVAILABLE;
    }
    public void init(Context context) {}
    public void initThermalPrediction(Context context, int time, int interval) {}
    public void pauseSession(Context context, String name) {}
    public void resumeSession(Context context, String name) {}
    public void startPrediction() {}
    public void startSession(Context context, String name) {}
    public void stopPrediction() {}
    public native void updatePrediction(float prediction);
}
