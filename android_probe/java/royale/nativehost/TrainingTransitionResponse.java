package royale.nativehost;

/** Envelopes trusted native JSON without constructing a second observation tree.
 * The operation name is distinct so old servers reject it before executing an
 * action. The client still parses and validates the complete JSON response.
 */
public final class TrainingTransitionResponse {
    private static final int MAX_FRAGMENT_CHARS = 32 * 1024 * 1024;

    private TrainingTransitionResponse() {}

    private static String objectFragment(String value, String name) {
        if (value == null || value.length() < 2 || value.length() > MAX_FRAGMENT_CHARS
                || value.charAt(0) != '{' || value.charAt(value.length() - 1) != '}') {
            throw new IllegalArgumentException("invalid native " + name + " fragment");
        }
        return value;
    }

    public static String encode(String actions, String episode, String state) {
        objectFragment(actions, "actions");
        objectFragment(episode, "episode");
        if (state != null) objectFragment(state, "state");
        long size = 160L + actions.length() + episode.length()
                + (state == null ? 0 : state.length());
        if (size > MAX_FRAGMENT_CHARS) {
            throw new IllegalArgumentException("native transition response too large");
        }
        StringBuilder result = new StringBuilder((int) size);
        result.append("{\"schema_version\":1,\"ok\":true,")
              .append("\"op\":\"joint_training_transition_fast_v1\",\"result\":{")
              .append("\"joint_action\":").append(actions)
              .append(",\"episode\":").append(episode);
        if (state != null) result.append(",\"state\":").append(state);
        return result.append("}}").toString();
    }
}
