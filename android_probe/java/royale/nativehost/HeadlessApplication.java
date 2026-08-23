package royale.nativehost;

import android.app.Application;
import android.content.Context;

/** Application instance attached only to the package Context, without a process component. */
public final class HeadlessApplication extends Application {
    public HeadlessApplication(Context context) {
        attachBaseContext(context);
    }
}
