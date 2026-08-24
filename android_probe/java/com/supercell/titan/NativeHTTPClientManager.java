package com.supercell.titan;

import java.util.Vector;

/** Offline HTTP bridge: preserves request IDs but opens no network sockets. */
public final class NativeHTTPClientManager {
    public static final Vector<Object> finishedConnections = new Vector<>();
    public static int idCounter = 1234;
    public static final NativeHTTPClientManager instance = new NativeHTTPClientManager();

    public static native void getFinished(
        boolean failed, int tag, byte[] data, int responseCode
    );
    public static native void postFinished(
        boolean failed, int tag, byte[] data, int responseCode
    );

    public static NativeHTTPClientManager getInstance() { return instance; }

    public static int startGetRequest(
        String url, String headers, String userAgent, String language
    ) {
        return idCounter++;
    }

    public static int startPostRequest(
        String url, String contentType, byte[] body,
        String[] headerNames, String[] headerValues
    ) {
        return idCounter++;
    }

    public void updateBeforeFrame() {}
}
