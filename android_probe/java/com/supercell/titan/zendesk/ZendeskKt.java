package com.supercell.titan.zendesk;

import android.content.Context;
import java.util.Map;

/** Offline customer-support bridge. */
public final class ZendeskKt {
    private ZendeskKt() {}

    public static void clearMetadata() {}
    public static void clearTags() {}
    public static int getUnreadMessageCount() { return 0; }
    public static void initializeMessagingSdk(
        Context context, String key, IMessagingSdkInitCallback callback
    ) {
        if (callback != null) { callback.onFailure(); }
    }
    public static void initializeSupportSdk(
        Context context, String url, String appId, String clientId
    ) {}
    public static void login(String jwt, IMessagingSdkLoginCallback callback) {
        if (callback != null) { callback.onFailure(); }
    }
    public static void setMetadata(String[] keys, String[] values) {}
    public static void setPushNotificationToken(String token) {}
    public static void setTags(String[] tags) {}
    public static void showHelpCenter(Context context) {}
    public static void showHelpCenterArticle(Context context, String article) {}
    public static void showMessaging(Context context) {}
    public static void showMessaging(Context context, String conversationId) {}
    public static boolean tryConsumePushNotification(
        Context context, Map<?, ?> data, boolean fromBackground
    ) { return false; }
}
