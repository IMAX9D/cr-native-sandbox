package com.supercell.titan;

/**
 * Headless replacement for the production Quago integration.
 *
 * <p>The battle engine references this class from its Android platform glue,
 * but device-attestation and telemetry are neither inputs nor outputs of the
 * deterministic battle simulation.  Loading the APK implementation from a
 * shell-owned process starts the protected Quago bootstrap and terminates the
 * process.  Keeping the public/JNI contract here lets libg use its normal
 * call-sites while the isolated host deliberately supplies no telemetry.</p>
 */
public final class QuagoManager {
    public QuagoManager(GameApp gameApp) {}

    public static void beginSegment(String name) {}

    public static void enable(boolean enabled, int flavor, boolean disableJsonCallback) {}

    public static void endSegment() {}

    public static native void onLog(int level, String tag, String message);

    public static void onPause() {}

    public static void onResume() {}

    public static native void sendJsonSegments(String segment, byte[] compressedJson);

    public static void setAdditionalId(String value) {}

    public static void setAppToken(String value) {}

    public static void setKeyValue(String key, String value) {}
}
