package com.supercell.titan;

import java.util.Locale;
import java.util.Map;
import java.util.TimeZone;
import java.util.concurrent.ConcurrentHashMap;

/** Headless, deterministic replacement for UI/device utility callbacks. */
public class ApplicationUtilBase {
    public static final String PREFERENCES_NEW = "headless-native-host";
    public static final String PREFERENCES_OLD = "headless-native-host";
    public static String m_cachedAndroidId = "headless-native-host";
    public static int totalMemory = 0;

    private static final Map<String, String> values = new ConcurrentHashMap<>();

    public static boolean canOpenURL(String value) { return false; }
    public static void copyString(String value) {}
    public static String getAndroidID() { return m_cachedAndroidId; }
    public static String getAppVersion() { return "15.535.29"; }
    public static String getBundleID() { return "com.supercell.clashroyale"; }
    public static String getIMEI() { return ""; }
    public static String getKeyValue(String key) { return values.getOrDefault(key, ""); }
    public static String getLocaleCountry() {
        String country = Locale.getDefault().getCountry();
        return country.isEmpty() ? "US" : country.toUpperCase(Locale.ROOT);
    }
    public static String getOpenUDID() { return ""; }
    public static String getPlatformDetail(int detail) { return ""; }
    public static String getPreferredLanguage() {
        return Locale.getDefault().toLanguageTag();
    }
    public static int getResidentMemoryInMB() { return 0; }
    public static String getServerEnvironment() { return ""; }
    public static String getTimeZoneID() { return TimeZone.getDefault().getID(); }
    public static int getTotalMemory() { return totalMemory; }
    public static boolean isAmazonDeviceMessagingSupported() { return false; }
    public static boolean isLowLatencyDevice() { return true; }
    public static void openMarketURL() {}
    public static void openURL(String value) {}
    public static String pasteString() { return ""; }
    public static void removeKeyValue(String key) { values.remove(key); }
    public static void setKeepScreenOn(boolean enabled) {}
    public static void setRequestedOrientation() {}
    public static void storeKeyValue(String key, String value) {
        if (key != null) {
            values.put(key, value == null ? "" : value);
        }
    }
}
