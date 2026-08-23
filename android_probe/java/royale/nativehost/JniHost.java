package royale.nativehost;

import com.supercell.titan.GameApp;
import com.supercell.titan.TitanApplication;
import android.graphics.SurfaceTexture;
import android.view.Surface;
import java.io.ByteArrayOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.Method;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import org.json.JSONArray;
import org.json.JSONObject;

public final class JniHost {
    private JniHost() {}

    private static final int TRACE_SCHEMA_VERSION = 1;
    private static final int MAX_TRACE_STEPS = 64;
    private static final int MIN_TRACE_RESPONSE_BYTES = 64 * 1024;
    private static final int MAX_TRACE_RESPONSE_BYTES = 32 * 1024 * 1024;

    private static String bootstrapReplayCanonical = null;
    private static int bootstrapReplaySeed = Integer.MIN_VALUE;
    private static boolean bootstrapReplayAvailable = false;
    private static boolean terminalEpisodeLatched = false;
    private static SurfaceTexture lifecycleSurfaceTexture = null;
    private static Surface lifecycleSurface = null;

    private static native String nativeCreateGameMain(
        String libgPath,
        Object assets,
        Object activity,
        String dataDir,
        String cacheDir,
        String externalCacheDir,
        long availableBytes,
        int width,
        int height,
        int densityDpi,
        float xdpi,
        float ydpi,
        int graphicsApi,
        String externalFilesDir
    );
    private static native String nativeProbeRuntime(String libgPath);
    private static native String nativeProbePrerequisites(String libgPath);
    private static native String nativeInitGameMain(String libgPath);
    private static native String nativeInitResources(String libgPath);
    private static native String nativeInitManager(String libgPath);
    private static native String nativePumpManager(String libgPath);
    private static native String nativePumpDataTables(String libgPath);
    private static native String nativeLoadReplay(String libgPath, String replayJson);
    private static native String nativeRestartReplay(String libgPath, String replayJson);
    private static native String nativeStep(String libgPath, int steps);
    private static native String nativeStepTrace(
        String libgPath, int steps, int traceSchemaVersion,
        int maxResponseBytes
    );
    private static native String nativeObserve(String libgPath);
    private static native String nativeAct(
        String libgPath, int side, int deckIndex, int x, int y,
        int accountHi, int accountLo, boolean dryRun
    );
    private static native String nativeProbeGrid(
        String libgPath, int side, int deckIndex, int accountHi, int accountLo
    );

    public static void main(String[] args) {
        if (args.length < 1 || args.length > 3) {
            System.err.println("usage: royale.nativehost.JniHost /absolute/runtime/root [load|context|create|lifecycle|replay|serve|probe-baseline|probe-detach-surface|probe-null-surface|probe-no-surface|probe-create-only|probe-minimal|probe-direct] [replay.json|port]");
            System.exit(64);
        }
        String mode = args.length >= 2 ? args[1] : "load";
        String path = args[0] + "/libg.so";
        System.load(args[0] + "/libnative_host_bridge.so");
        System.out.println(
            "{\"schema_version\":1,\"stage\":\"jni_on_load\","
                + "\"event\":\"before_system_load\",\"path\":\""
                + path.replace("\\", "\\\\").replace("\"", "\\\"")
                + "\"}"
        );
        System.out.flush();
        try {
            System.load(path);
        } catch (Throwable error) {
            System.err.println("SYSTEM_LOAD_FAILED type=" + error.getClass().getName());
            error.printStackTrace(System.err);
            System.err.flush();
            System.exit(2);
        }
        System.out.println(
            "{\"schema_version\":1,\"stage\":\"jni_on_load\","
                + "\"event\":\"after_system_load\",\"ok\":true}"
        );
        System.out.flush();
        if ("load".equals(mode)) {
            emitRuntimeProbe(args[0], "after_system_load");
            return;
        }
        try {
            Object packageContext = createPackageContext("com.supercell.clashroyale");
            Method getAssets = Class.forName("android.content.Context").getMethod("getAssets");
            Object assets = getAssets.invoke(packageContext);
            System.out.println(
                "{\"schema_version\":1,\"stage\":\"package_context\","
                    + "\"ok\":true,\"context_class\":\""
                    + packageContext.getClass().getName()
                    + "\",\"assets_class\":\""
                    + assets.getClass().getName()
                    + "\"}"
            );
            System.out.flush();
            if ("context".equals(mode)) {
                return;
            }
            if (!"create".equals(mode) && !"lifecycle".equals(mode)
                && !"replay".equals(mode) && !"serve".equals(mode)
                && !isProbeMode(mode)) {
                throw new IllegalArgumentException("unknown mode: " + mode);
            }
            if (isLifecycleMode(mode)) {
                TitanApplication.bindContext((android.content.Context) packageContext);
                emitRuntimeProbe(args[0], "after_system_load");
                emitStage("native_libraries_loaded", "before");
                TitanApplication.nOnNativeLibrariesLoaded();
                emitStage("native_libraries_loaded", "after");
                emitRuntimeProbe(args[0], "after_native_libraries_loaded");
                emitStage("application_create", "before");
                GameApp.nOnApplicationCreate();
                emitStage("application_create", "after");
                emitRuntimeProbe(args[0], "after_application_create");
            }
            invokeCreateGameMain(args[0], assets, packageContext);
            if (isLifecycleMode(mode)) {
                emitRuntimeProbe(args[0], "after_create_game_main");
                if (usesActivityCreate(mode)) {
                    emitStage("activity_create", "before");
                    GameApp.nOnCreate();
                    emitStage("activity_create", "after");
                    emitRuntimeProbe(args[0], "after_activity_create");
                }
                if (usesSurface(mode)) {
                    emitStage("surface_create", "before");
                    lifecycleSurfaceTexture = new SurfaceTexture(false);
                    lifecycleSurfaceTexture.setDefaultBufferSize(1080, 2400);
                    lifecycleSurface = new Surface(lifecycleSurfaceTexture);
                    GameApp.nOnSurfaceCreated(lifecycleSurface);
                    GameApp.nOnSurfaceChanged(lifecycleSurface, 1080, 2400);
                    emitStage("surface_create", "after");
                    emitRuntimeProbe(args[0], "after_surface_create");
                }
                if ("probe-null-surface".equals(mode)) {
                    emitStage("null_surface_callback", "before");
                    GameApp.nOnSurfaceCreated(null);
                    GameApp.nOnSurfaceChanged(null, 1080, 2400);
                    emitStage("null_surface_callback", "after");
                    emitRuntimeProbe(args[0], "after_null_surface_callback");
                }
                if (usesStartResume(mode)) {
                    emitStage("activity_start", "before");
                    GameApp.nOnStart();
                    emitStage("activity_start", "after");
                    emitRuntimeProbe(args[0], "after_activity_start");
                    emitStage("activity_resume", "before");
                    GameApp.nOnResume();
                    emitStage("activity_resume", "after");
                    emitRuntimeProbe(args[0], "after_activity_resume");
                }
                emitStage("headless_hold", "before");
                Thread.sleep(5000L);
                emitStage("headless_hold", "after");
                emitRuntimeProbe(args[0], "after_headless_hold");
                System.out.println(
                    "{\"schema_version\":1,\"stage\":\"prerequisite_probe\"," +
                        "\"event\":\"after_headless_hold\",\"value\":" +
                        nativeProbePrerequisites(args[0] + "/libg.so") + "}"
                );
                System.out.flush();
                if ("probe-direct".equals(mode)) {
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"direct_platform_init\"," +
                            "\"event\":\"after_call\",\"value\":" +
                            nativeInitResources(args[0] + "/libg.so") + "}"
                    );
                    System.out.flush();
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"direct_game_main_init\"," +
                            "\"event\":\"after_call\",\"value\":" +
                            nativeInitGameMain(args[0] + "/libg.so") + "}"
                    );
                    System.out.flush();
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"direct_loading_state\"," +
                            "\"event\":\"after_call\",\"value\":" +
                            nativePumpManager(args[0] + "/libg.so") + "}"
                    );
                    System.out.flush();
                    JSONObject directLoadingFrame = null;
                    for (int loadingFrame = 0; loadingFrame < 8;
                         ++loadingFrame) {
                        directLoadingFrame = new JSONObject(
                            nativePumpManager(args[0] + "/libg.so")
                        );
                    }
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"direct_loading_frames\"," +
                            "\"event\":\"after_call\",\"value\":" +
                            directLoadingFrame.toString() + "}"
                    );
                    String preDataTablesJson =
                        nativeProbePrerequisites(args[0] + "/libg.so");
                    JSONObject preDataTables = new JSONObject(
                        preDataTablesJson
                    );
                    if ("0x0".equals(preDataTables.optString(
                            "battle_data_content", "0x0")) &&
                        !"0x0".equals(preDataTables.optString(
                            "data_load_task", "0x0"))) {
                        System.out.println(
                            "{\"schema_version\":1,\"stage\":\"direct_data_tables\"," +
                                "\"event\":\"after_call\",\"value\":" +
                                nativePumpDataTables(args[0] + "/libg.so") + "}"
                        );
                        JSONObject postDataTablesFrame = null;
                        for (int finalizeFrame = 0; finalizeFrame < 4;
                             ++finalizeFrame) {
                            postDataTablesFrame = new JSONObject(
                                nativePumpManager(args[0] + "/libg.so")
                            );
                        }
                        System.out.println(
                            "{\"schema_version\":1," +
                                "\"stage\":\"direct_data_tables_finalize\"," +
                                "\"event\":\"after_call\",\"value\":" +
                                postDataTablesFrame.toString() + "}"
                        );
                    }
                    System.out.flush();
                    String directReadinessJson =
                        nativeProbePrerequisites(args[0] + "/libg.so");
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"prerequisite_probe\"," +
                            "\"event\":\"after_direct_resources\",\"value\":" +
                            directReadinessJson + "}"
                    );
                    System.out.flush();
                    emitRuntimeProbe(args[0], "after_direct_manager_init");
                    JSONObject directReadiness = new JSONObject(
                        directReadinessJson
                    );
                    if ("0x0".equals(
                            directReadiness.optString(
                                "battle_data_content", "0x0"
                            ))) {
                        JSONObject blocked = new JSONObject();
                        blocked.put("profile", mode);
                        blocked.put("status", "blocked_data_tables");
                        blocked.put("game_main_initialized", true);
                        blocked.put("manager_initialized", true);
                        blocked.put("surface_created", false);
                        blocked.put(
                            "blocker",
                            "native DataTables container exists but its table " +
                                "array has not been populated"
                        );
                        blocked.put("readiness", directReadiness);
                        System.out.println(
                            "{\"schema_version\":1,\"stage\":\"probe_result\"," +
                                "\"event\":\"blocked\",\"value\":" +
                                blocked.toString() + "}"
                        );
                        System.out.flush();
                        System.exit(0);
                    }
                }
                if ("serve".equals(mode)) {
                    String bootstrapReplay = readUtf8(
                        new File(args[0], "bootstrap-replay.json")
                    );
                    JSONObject bootstrapReplayObject = new JSONObject(
                        bootstrapReplay
                    );
                    bootstrapReplayCanonical = bootstrapReplayObject.toString();
                    bootstrapReplaySeed = bootstrapReplayObject.optInt(
                        "rndSeed", Integer.MIN_VALUE
                    );
                    emitStage("service_bootstrap", "before");
                    nativeLoadReplay(args[0] + "/libg.so", bootstrapReplay);
                    waitForBattle(args[0], 5000L, null, 0);
                    emitStage("service_bootstrap", "after");
                    GameApp.nOnPause();
                    Thread.sleep(100L);
                    JSONObject bootstrapStep = new JSONObject(
                        nativeStep(args[0] + "/libg.so", 10)
                    );
                    if (bootstrapStep.optInt("tick_after", -1) < 5) {
                        throw new IllegalStateException(
                            "controlled bootstrap did not reach tick 5: "
                                + bootstrapStep.toString()
                        );
                    }
                    bootstrapReplayAvailable = true;
                    emitStage("controlled_clock", "paused");
                }
                if (isProbeMode(mode)) {
                    if (args.length != 3) {
                        throw new IllegalArgumentException(
                            mode + " requires a replay JSON path"
                        );
                    }
                    String replayJson = readUtf8(new File(args[2]));
                    long startedNanos = System.nanoTime();
                    JSONObject loadResult = new JSONObject(
                        nativeLoadReplay(args[0] + "/libg.so", replayJson)
                    );
                    JSONObject pumpResult = null;
                    if ("probe-direct".equals(mode)) {
                        pumpResult = new JSONObject(
                            nativePumpManager(args[0] + "/libg.so")
                        );
                    }
                    JSONObject readyState = waitForBattle(
                        args[0], 5000L, null, 0
                    );
                    if (usesStartResume(mode)) {
                        GameApp.nOnPause();
                        Thread.sleep(100L);
                    }
                    if ("probe-detach-surface".equals(mode)) {
                        GameApp.nOnSurfaceDestroyed(lifecycleSurface);
                        lifecycleSurface.release();
                        lifecycleSurfaceTexture.release();
                        lifecycleSurface = null;
                        lifecycleSurfaceTexture = null;
                        emitStage("surface_detached", "after");
                    }
                    JSONObject stepResult = new JSONObject(
                        nativeStep(args[0] + "/libg.so", 100)
                    );
                    JSONObject finalState = new JSONObject(
                        nativeObserve(args[0] + "/libg.so")
                    );
                    long elapsedNanos = System.nanoTime() - startedNanos;
                    JSONObject result = new JSONObject();
                    result.put("profile", mode);
                    result.put("load", loadResult);
                    if (pumpResult != null) {
                        result.put("pump", pumpResult);
                    }
                    result.put("ready", readyState);
                    result.put("step", stepResult);
                    result.put("state", finalState);
                    result.put("elapsed_ms", elapsedNanos / 1000000.0);
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"probe_result\"," +
                            "\"event\":\"complete\",\"value\":" +
                            result.toString() + "}"
                    );
                    System.out.flush();
                } else if ("replay".equals(mode)) {
                    if (args.length != 3) {
                        throw new IllegalArgumentException("replay mode requires a JSON path");
                    }
                    String replayJson = readUtf8(new File(args[2]));
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"replay_input\","
                            + "\"event\":\"after_call\",\"value\":"
                            + nativeLoadReplay(args[0] + "/libg.so", replayJson) + "}"
                    );
                    System.out.flush();
                    JSONObject readyState = waitForBattle(
                        args[0], 3000L, null, 5
                    );
                    GameApp.nOnPause();
                    Thread.sleep(100L);
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"controlled_clock\","
                            + "\"event\":\"paused\",\"value\":"
                            + readyState.toString() + "}"
                    );
                    System.out.println(
                        "{\"schema_version\":1,\"stage\":\"controlled_step\","
                            + "\"event\":\"after_call\",\"value\":"
                            + nativeStep(args[0] + "/libg.so", 1) + "}"
                    );
                    System.out.flush();
                    emitRuntimeProbe(args[0], "after_replay_input");
                } else if ("serve".equals(mode)) {
                    int port = args.length == 3 ? Integer.parseInt(args[2]) : 37031;
                    serveJson(args[0], port);
                }
                emitStage("activity_destroy", "before");
                if (usesStartResume(mode)) {
                    GameApp.nOnPause();
                    GameApp.nOnStop();
                }
                if (usesSurface(mode) && lifecycleSurface != null) {
                    GameApp.nOnSurfaceDestroyed(lifecycleSurface);
                    lifecycleSurface.release();
                    lifecycleSurfaceTexture.release();
                }
                lifecycleSurface = null;
                lifecycleSurfaceTexture = null;
                if (usesActivityCreate(mode)) {
                    GameApp.nOnDestroy();
                }
                emitStage("activity_destroy", "after");
                System.exit(0);
            }
        } catch (Throwable error) {
            System.err.println("NATIVE_STAGE_FAILED mode=" + mode + " type=" + error.getClass().getName());
            error.printStackTrace(System.err);
            System.err.flush();
            System.exit(3);
        }
    }

    private static JSONObject restartBattleLifecycle(
        String root, String replay
    ) throws Exception {
        JSONObject current = new JSONObject(
            nativeProbeRuntime(root + "/libg.so")
        );
        int currentTick = current.optInt("tick", -1);
        if (!terminalEpisodeLatched && currentTick >= 0 && currentTick < 1000) {
            throw new IllegalStateException(
                "native replay restart requires tick >= 1000; recycle the "
                    + "host for an opening-countdown reset"
            );
        }
        JSONObject loaded = new JSONObject(
            nativeRestartReplay(root + "/libg.so", replay)
        );
        // CE7810 synchronously constructs and enters the replacement native
        // BattleGameState.  The service clock is already paused; resuming the
        // Android lifecycle here creates a second manager and falls back to
        // HomeState.  Keep the new battle under nativeStep control instead.
        JSONObject readyState = waitForBattle(
            root, 5000L, null, 0
        );
        JSONObject warmup = null;
        for (int attempt = 0;
             attempt < 4 && readyState.optInt("tick", -1) < 10;
             ++attempt) {
            int remaining = Math.max(
                1, 10 - readyState.optInt("tick", 0)
            );
            warmup = new JSONObject(
                nativeStep(root + "/libg.so", remaining)
            );
            readyState = new JSONObject(
                nativeProbeRuntime(root + "/libg.so")
            );
        }
        if (readyState.optInt("tick", -1) < 10) {
            throw new IllegalStateException(
                "lifecycle restart warmup failed: "
                    + String.valueOf(warmup)
            );
        }
        terminalEpisodeLatched = false;
        bootstrapReplayAvailable = false;
        JSONObject result = new JSONObject();
        result.put("lifecycle_restarted", true);
        result.put("load", loaded);
        result.put("state", readyState);
        return result;
    }

    private static void emitStage(String stage, String event) {
        System.out.println(
            "{\"schema_version\":1,\"stage\":\"" + stage
                + "\",\"event\":\"" + event + "\"}"
        );
        System.out.flush();
    }

    private static boolean isLifecycleMode(String mode) {
        return "lifecycle".equals(mode) || "replay".equals(mode)
            || "serve".equals(mode) || isProbeMode(mode);
    }

    private static boolean isProbeMode(String mode) {
        return "probe-baseline".equals(mode)
            || "probe-detach-surface".equals(mode)
            || "probe-null-surface".equals(mode)
            || "probe-no-surface".equals(mode)
            || "probe-create-only".equals(mode)
            || "probe-minimal".equals(mode)
            || "probe-direct".equals(mode);
    }

    private static boolean usesActivityCreate(String mode) {
        return !"probe-minimal".equals(mode);
    }

    private static boolean usesSurface(String mode) {
        return !isProbeMode(mode) || "probe-baseline".equals(mode)
            || "probe-detach-surface".equals(mode);
    }

    private static boolean usesStartResume(String mode) {
        return !isProbeMode(mode) || "probe-baseline".equals(mode)
            || "probe-detach-surface".equals(mode)
            || "probe-null-surface".equals(mode)
            || "probe-no-surface".equals(mode);
    }

    private static void serveJson(String root, int port) throws Exception {
        InetAddress loopback = InetAddress.getByName("127.0.0.1");
        try (ServerSocket server = new ServerSocket(port, 16, loopback)) {
            System.out.println(
                "{\"schema_version\":1,\"stage\":\"json_server\","
                    + "\"event\":\"ready\",\"address\":\"127.0.0.1\","
                    + "\"port\":" + port + "}"
            );
            System.out.flush();
            boolean running = true;
            while (running) {
                try (Socket socket = server.accept();
                     BufferedReader reader = new BufferedReader(new InputStreamReader(
                         socket.getInputStream(), StandardCharsets.UTF_8
                     ));
                     BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                         socket.getOutputStream(), StandardCharsets.UTF_8
                     ))) {
                    JSONObject response = new JSONObject();
                    try {
                        String line = reader.readLine();
                        if (line == null || line.length() > 32 * 1024 * 1024) {
                            throw new IllegalArgumentException("invalid JSON line length");
                        }
                        JSONObject request = new JSONObject(line);
                        String op = request.getString("op");
                        response.put("schema_version", 1);
                        response.put("ok", true);
                        response.put("op", op);
                        if ("status".equals(op)) {
                            response.put(
                                "state", new JSONObject(nativeProbeRuntime(root + "/libg.so"))
                            );
                        } else if ("restart_replay".equals(op)) {
                            throw new UnsupportedOperationException(
                                "restart_replay is disabled; start a fresh "
                                    + "native service with a seeded bootstrap"
                            );
                        } else if ("load_replay".equals(op)) {
                            if (terminalEpisodeLatched) {
                                throw new IllegalStateException(
                                    "native terminal is latched; recycle the host process before reset"
                                );
                            }
                            JSONObject previousState = new JSONObject(
                                nativeProbeRuntime(root + "/libg.so")
                            );
                            String previousReplay = previousState.optString(
                                "replay_data", null
                            );
                            JSONObject readyState;
                            try {
                                String replay = request.getJSONObject("replay").toString();
                                boolean adoptBootstrap = bootstrapReplayAvailable
                                    && (replay.equals(bootstrapReplayCanonical)
                                        || request.getJSONObject("replay").optInt(
                                            "rndSeed", Integer.MAX_VALUE
                                        ) == bootstrapReplaySeed)
                                    && previousState.optInt(
                                        "current_state_type", -1
                                    ) == 4;
                                bootstrapReplayAvailable = false;
                                if (adoptBootstrap) {
                                    JSONObject adopted = new JSONObject();
                                    adopted.put("called", false);
                                    adopted.put("adopted_bootstrap", true);
                                    response.put("result", adopted);
                                    // The bootstrap is already paused and was
                                    // advanced by the controlled native core.
                                    // Resuming here creates a one-frame race in
                                    // which the presentation state can discard
                                    // the headless battle before nOnPause lands.
                                    readyState = previousState;
                                } else {
                                    response.put(
                                        "result", new JSONObject(
                                            nativeLoadReplay(root + "/libg.so", replay)
                                        )
                                    );
                                    GameApp.nOnResume();
                                    readyState = waitForBattle(
                                        root, 3000L, previousReplay, 0
                                    );
                                }
                            } finally {
                                GameApp.nOnPause();
                                Thread.sleep(100L);
                            }
                            if (readyState.optInt("tick", -1) < 5) {
                                JSONObject controlledWarmup = new JSONObject(
                                    nativeStep(root + "/libg.so", 10)
                                );
                                if (controlledWarmup.optInt("tick_after", -1) < 5) {
                                    throw new IllegalStateException(
                                        "controlled replay warmup failed: "
                                            + controlledWarmup.toString()
                                    );
                                }
                                readyState = new JSONObject(
                                    nativeProbeRuntime(root + "/libg.so")
                                );
                            }
                            response.put(
                                "state", readyState
                            );
                        } else if ("step".equals(op)) {
                            int steps = request.optInt("steps", 1);
                            JSONObject stepResult = new JSONObject(
                                nativeStep(root + "/libg.so", steps)
                            );
                            response.put("result", stepResult);
                            terminalEpisodeLatched = stepResult
                                .getJSONObject("episode")
                                .optBoolean("terminated", false);
                        } else if ("step_trace".equals(op)) {
                            JSONObject traceResult = executeStepTrace(root, request);
                            response.put("result", traceResult);
                            terminalEpisodeLatched = traceResult.optBoolean(
                                "terminal", false
                            );
                        } else if ("joint_transition_trace".equals(op)) {
                            JSONObject result = new JSONObject();
                            result.put(
                                "joint_action",
                                executeJointActions(
                                    root, request.getJSONArray("actions")
                                )
                            );
                            JSONObject traceResult = executeStepTrace(root, request);
                            result.put("trace", traceResult);
                            result.put("episode", finalTraceEpisode(traceResult));
                            response.put("result", result);
                            terminalEpisodeLatched = traceResult.optBoolean(
                                "terminal", false
                            );
                        } else if ("observe".equals(op)) {
                            response.put(
                                "state", new JSONObject(
                                    nativeObserve(root + "/libg.so")
                                )
                            );
                        } else if ("probe_grid".equals(op)) {
                            JSONObject action = request.getJSONObject("action");
                            int side = action.getInt("side");
                            response.put(
                                "result",
                                new JSONObject(
                                    nativeProbeGrid(
                                        root + "/libg.so", side,
                                        action.getInt("deck_index"),
                                        action.optInt("account_hi", side + 1),
                                        action.optInt("account_lo", side + 1)
                                    )
                                )
                            );
                        } else if ("joint_act".equals(op)) {
                            response.put(
                                "result",
                                executeJointActions(
                                    root, request.getJSONArray("actions")
                                )
                            );
                        } else if ("joint_transition".equals(op)) {
                            JSONObject result = new JSONObject();
                            result.put(
                                "joint_action",
                                executeJointActions(
                                    root, request.getJSONArray("actions")
                                )
                            );
                            int steps = request.optInt("steps", 1);
                            JSONObject stepResult = new JSONObject(
                                nativeStep(root + "/libg.so", steps)
                            );
                            result.put("step", stepResult);
                            JSONObject episode = stepResult.getJSONObject("episode");
                            terminalEpisodeLatched = episode.optBoolean(
                                "terminated", false
                            );
                            if (!terminalEpisodeLatched
                                && !episode.optBoolean("truncated", false)) {
                                result.put(
                                    "state",
                                    new JSONObject(nativeObserve(root + "/libg.so"))
                                );
                            }
                            response.put("result", result);
                        } else if ("act".equals(op)) {
                            JSONObject action = request.getJSONObject("action");
                            response.put("result", executeAction(root, action));
                        } else if ("shutdown".equals(op)) {
                            running = false;
                        } else if (!"ping".equals(op)) {
                            throw new IllegalArgumentException("unknown op: " + op);
                        }
                    } catch (Throwable error) {
                        response = new JSONObject();
                        response.put("schema_version", 1);
                        response.put("ok", false);
                        response.put("error_type", error.getClass().getName());
                        response.put("error", String.valueOf(error.getMessage()));
                    }
                    writer.write(response.toString());
                    writer.newLine();
                    writer.flush();
                }
            }
        }
    }

    private static JSONObject executeStepTrace(
        String root, JSONObject request
    ) throws Exception {
        int traceSchemaVersion = request.optInt("trace_schema_version", -1);
        if (traceSchemaVersion != TRACE_SCHEMA_VERSION) {
            throw new IllegalArgumentException(
                "step_trace requires trace_schema_version=1"
            );
        }
        int steps = request.getInt("steps");
        if (steps < 1 || steps > MAX_TRACE_STEPS) {
            throw new IllegalArgumentException(
                "step_trace steps must be in 1..64"
            );
        }
        int maxResponseBytes = request.optInt(
            "max_response_bytes", MAX_TRACE_RESPONSE_BYTES
        );
        if (maxResponseBytes < MIN_TRACE_RESPONSE_BYTES
            || maxResponseBytes > MAX_TRACE_RESPONSE_BYTES) {
            throw new IllegalArgumentException(
                "step_trace max_response_bytes must be in 65536..33554432"
            );
        }
        String traceJson = nativeStepTrace(
            root + "/libg.so", steps, traceSchemaVersion, maxResponseBytes
        );
        if (traceJson.getBytes(StandardCharsets.UTF_8).length
            > maxResponseBytes) {
            throw new IllegalStateException(
                "native step_trace exceeded response limit"
            );
        }
        JSONObject traceResult = new JSONObject(traceJson);
        JSONArray frames = traceResult.getJSONArray("frames");
        int stepped = traceResult.getInt("stepped");
        JSONObject initialFrame = traceResult.getJSONObject("initial_frame");
        if (traceResult.getInt("schema_version") != 1
            || traceResult.getInt("trace_schema_version")
                != TRACE_SCHEMA_VERSION
            || !"libg_native_tick_trace".equals(traceResult.getString("kind"))
            || !"full-v1".equals(traceResult.getString("encoding"))
            || traceResult.getInt("requested_steps") != steps
            || traceResult.getInt("max_response_bytes") != maxResponseBytes
            || stepped < 0 || stepped > steps || frames.length() != stepped
            || traceResult.getInt("final_frame_index") != stepped
            || initialFrame.getInt("frame_index") != 0
            || initialFrame.getInt("advanced_steps") != 0
            || !initialFrame.has("observation_complete")
            || !initialFrame.has("state")) {
            throw new IllegalStateException(
                "native step_trace contract mismatch"
            );
        }
        for (int index = 0; index < frames.length(); ++index) {
            JSONObject frame = frames.getJSONObject(index);
            if (frame.getInt("frame_index") != index + 1
                || frame.getInt("advanced_steps") != index + 1
                || !frame.has("observation_complete")
                || !frame.has("step") || !frame.has("state")) {
                throw new IllegalStateException(
                    "native step_trace frame mismatch"
                );
            }
        }
        if (traceResult.optBoolean("terminal", false)
            && !finalTraceEpisode(traceResult).optBoolean("terminated", false)) {
            throw new IllegalStateException(
                "native step_trace terminal frame mismatch"
            );
        }
        return traceResult;
    }

    private static JSONObject finalTraceEpisode(JSONObject traceResult)
        throws Exception {
        int stepped = traceResult.getInt("stepped");
        JSONObject finalFrame = stepped == 0
            ? traceResult.getJSONObject("initial_frame")
            : traceResult.getJSONArray("frames").getJSONObject(stepped - 1);
        return finalFrame.getJSONObject("state").getJSONObject("episode");
    }

    private static JSONObject executeAction(
        String root, JSONObject action
    ) throws Exception {
        return new JSONObject(
            nativeAct(
                root + "/libg.so",
                action.getInt("side"),
                action.getInt("deck_index"),
                action.getInt("x"),
                action.getInt("y"),
                action.optInt("account_hi", action.getInt("side") + 1),
                action.optInt("account_lo", action.getInt("side") + 1),
                action.optBoolean("dry_run", false)
            )
        );
    }

    private static JSONObject executeJointActions(
        String root, JSONArray requested
    ) throws Exception {
        if (requested.length() > 2) {
            throw new IllegalArgumentException(
                "joint action accepts at most one action per side"
            );
        }
        boolean[] seen = {false, false};
        for (int index = 0; index < requested.length(); ++index) {
            int side = requested.getJSONObject(index).getInt("side");
            if (side < 0 || side > 1 || seen[side]) {
                throw new IllegalArgumentException(
                    "joint action requires unique side 0 and/or 1"
                );
            }
            seen[side] = true;
        }
        JSONArray applied = new JSONArray();
        for (int side = 0; side <= 1; ++side) {
            JSONObject selected = null;
            for (int index = 0; index < requested.length(); ++index) {
                JSONObject candidate = requested.getJSONObject(index);
                if (candidate.getInt("side") == side) {
                    if (selected != null) {
                        throw new IllegalArgumentException(
                            "duplicate joint action side: " + side
                        );
                    }
                    selected = candidate;
                }
            }
            if (selected != null) {
                JSONObject item = new JSONObject();
                item.put("side", side);
                item.put("result", executeAction(root, selected));
                applied.put(item);
            }
        }
        JSONObject result = new JSONObject();
        result.put("canonical_order", "side_0_then_side_1");
        result.put("actions", applied);
        return result;
    }

    private static String readUtf8(File path) throws IOException {
        try (FileInputStream input = new FileInputStream(path);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[16 * 1024];
            for (int count; (count = input.read(chunk)) >= 0; ) {
                output.write(chunk, 0, count);
            }
            return new String(output.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private static JSONObject waitForBattle(
        String root, long timeoutMillis, String previousReplay, int minimumTick
    )
        throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        JSONObject last = null;
        while (System.currentTimeMillis() < deadline) {
            last = new JSONObject(nativeProbeRuntime(root + "/libg.so"));
            if (last.optInt("current_state_type", -1) == 4
                && last.optInt("pending_state_type", -1) == 0
                && last.optInt("tick", -1) >= minimumTick
                && !"0x0".equals(last.optString("battle", "0x0"))
                && (previousReplay == null
                    || !previousReplay.equals(
                        last.optString("replay_data", "0x0")
                    ))) {
                return last;
            }
            Thread.sleep(2L);
        }
        throw new IllegalStateException(
            "native battle did not become ready: " + String.valueOf(last)
        );
    }

    private static void emitRuntimeProbe(String root, String event) {
        System.out.println(
            "{\"schema_version\":1,\"stage\":\"runtime_probe\","
                + "\"event\":\"" + event + "\",\"value\":"
                + nativeProbeRuntime(root + "/libg.so") + "}"
        );
        System.out.flush();
    }

    private static Object createPackageContext(String packageName) throws Exception {
        Class<?> looperClass = Class.forName("android.os.Looper");
        Method myLooper = looperClass.getMethod("myLooper");
        if (myLooper.invoke(null) == null) {
            looperClass.getMethod("prepareMainLooper").invoke(null);
        }
        Class<?> activityThreadClass = Class.forName("android.app.ActivityThread");
        Method systemMain = activityThreadClass.getDeclaredMethod("systemMain");
        systemMain.setAccessible(true);
        Object activityThread = systemMain.invoke(null);
        Method getSystemContext = activityThreadClass.getDeclaredMethod("getSystemContext");
        getSystemContext.setAccessible(true);
        Object systemContext = getSystemContext.invoke(activityThread);
        Class<?> contextClass = Class.forName("android.content.Context");
        Method createPackageContext = contextClass.getMethod(
            "createPackageContext", String.class, int.class
        );
        // CONTEXT_INCLUDE_CODE | CONTEXT_IGNORE_SECURITY. The process remains
        // shell-owned and receives no access to the game's private data.
        return createPackageContext.invoke(systemContext, packageName, 3);
    }

    private static void invokeCreateGameMain(
        String root, Object assets, Object packageContext
    ) throws Exception {
        for (String name : new String[] {"data", "cache", "external"}) {
            File directory = new File(root, name);
            if (!directory.exists() && !directory.mkdirs()) {
                throw new IllegalStateException("cannot create " + directory);
            }
        }
        GameApp activity = new GameApp(
            (android.content.Context) packageContext
        );
        String result = nativeCreateGameMain(
            root + "/libg.so",
            assets,
            activity,
            root + "/data",
            root + "/cache",
            root + "/external",
            8L * 1024L * 1024L * 1024L,
            1080,
            2400,
            420,
            420.0f,
            420.0f,
            2,
            root + "/external"
        );
        System.out.println(
            "{\"schema_version\":1,\"stage\":\"create_game_main\","
                + "\"ok\":true,\"result\":\""
                + String.valueOf(result).replace("\\", "\\\\").replace("\"", "\\\"")
                + "\"}"
        );
        System.out.flush();
    }
}
