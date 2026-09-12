import royale.nativehost.TrainingTransitionResponse;

public class TrainingTransitionResponseHarness {
    public static void main(String[] args) {
        System.out.println(TrainingTransitionResponse.encode(
            args[0], args[1], args.length > 2 ? args[2] : null
        ));
    }
}
