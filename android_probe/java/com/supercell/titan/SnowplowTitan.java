package com.supercell.titan;

import org.json.JSONObject;
import java.util.HashMap;

/** Deterministic no-network analytics bridge for native initialization. */
public final class SnowplowTitan {
    private SnowplowTitan() {}

    public static void createTracker(
        String namespace, String endpoint, String appId,
        boolean base64, boolean sessionTracking
    ) {}

    public static String getInstallationUserId() { return "headless-native-host"; }
    public static String getSessionId() { return ""; }
    public static HashMap<String, Object> jsonToMap(JSONObject value) {
        return new HashMap<>();
    }
    public static void removeAllTrackers() {}
    public static void setUserId(String userId) {}
    public static void trackSelfDescribingEvent(
        String schema, String data, String[] contextSchemas, String[] contextData
    ) {}
    public static void trackStructuredEvent(
        String category, String action, String label, String property,
        double value, boolean hasValue
    ) {}
}
