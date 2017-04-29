package deepwater.backends.tensorflow.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class TFPythonWrapper {
    private static final String GEN_SCRIPT = "h2o_deepwater_generate_models.py";
    private static final Map<String, String> GEN_FILES = new HashMap<String, String>() {{
        this.put(GEN_SCRIPT, GEN_SCRIPT);
        this.put("train.py", "deepwater/");
        this.put("optimizers.py", "deepwater/");
        this.put("alexnet.py", "deepwater/models/");
        this.put("inception.py", "deepwater/models/");
        this.put("lenet.py", "deepwater/models/");
        this.put("mlp.py", "deepwater/models/");
        this.put("nn.py", "deepwater/models/");
        this.put("resnet.py", "deepwater/models/");
        this.put("utils.py", "deepwater/models/");
        this.put("vgg.py", "deepwater/models/");
    }};

    public static String generateMetaFile(
            String networkType,
            int width,
            int height,
            int channels,
            int numClasses,
            int[] hiddens) {

        templateFilePresent(networkType);

        try {
            tensorflowInstalled();

            File output = File.createTempFile(System.getProperty("java.io.tmpdir"), Long.toString(System.nanoTime()));

            String runScript = extractGenFiles(output);

            return runGenScript(output, runScript, networkType, width, height, channels, numClasses, hiddens);
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException(e);
        }
    }

    private static void templateFilePresent(String networkType) {
        if(null == TFPythonWrapper.class.getResource(networkType + ".py")) {
            throw new IllegalArgumentException("Tensorflow template file for network [" + networkType + "] could not be found. Please check the documentation" +
                    " for a list of currently supported networks.");
        }
    }

    private static void tensorflowInstalled() throws IOException, InterruptedException {
        // TODO does this work?
        if(Runtime.getRuntime().exec("python -c 'import tensorflow'").waitFor() != 0){
            throw new IllegalArgumentException("Python tensorflow not installed on this machine. Please run 'pip install tensorflow[-gpu]' first.");
        }
    }

    private static String extractGenFiles(File output) throws IOException {
        for(Map.Entry<String, String> entry : GEN_FILES.entrySet()) {
            Files.copy(
                    TFPythonWrapper.class.getResource(entry.getKey()).openStream(),
                    new File(output, entry.getValue()).toPath()
            );
        }

        return output.getAbsolutePath() + File.pathSeparator + GEN_SCRIPT;
    }

    private static String runGenScript(File output,
                                       String runScript,
                                       String networkType,
                                       int width,
                                       int height,
                                       int channels,
                                       int numClasses,
                                       int[] hiddens) throws IOException {
        StringBuilder command = new StringBuilder("python "
                + runScript + " "
                + output.getAbsolutePath() + " "
                + networkType + " "
                + width + " "
                + height + " "
                + channels + " "
                + numClasses + " ");

        if(null != hiddens) {
            command.append(Arrays.toString(hiddens));
        }

        Runtime.getRuntime().exec(command.toString());
        return output.getAbsolutePath() + File.pathSeparator +
                networkType + "_"
                + width + "x"
                + height + "x"
                + channels + "_"
                + numClasses
                + ".meta";
    }
}
