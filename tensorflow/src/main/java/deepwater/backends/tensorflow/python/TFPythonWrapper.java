package deepwater.backends.tensorflow.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class TFPythonWrapper {
    private static final String GEN_SCRIPT = "h2o_deepwater_generate_models.py";
    private static final Map<String, String> GEN_FILES = new HashMap<String, String>() {{
        this.put(GEN_SCRIPT, "");
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
        this.put("__init__.py", "deepwater/models/");
    }};

    public static String generateMetaFile(
            String networkType,
            int width,
            int height,
            int channels,
            int numClasses,
            int[] hiddens) {

        networkType = networkType.toLowerCase();
        if(networkType.endsWith("_bn")) {
            networkType = networkType.replace("_bn", "");
        }

        templateFilePresent(networkType);

        try {
            tensorflowInstalled();

            Path output = Files.createTempDirectory(Paths.get(System.getProperty("java.io.tmpdir")), Long.toString(System.nanoTime()));

            String runScript = extractGenFiles(output);

            return runGenScript(output, runScript, networkType, width, height, channels, numClasses, hiddens);
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException(e);
        }
    }

    private static void templateFilePresent(String networkType) {
        if(null == TFPythonWrapper.class.getResource("/" + networkType + ".py")) {
            throw new IllegalArgumentException("Tensorflow template file for network [" + networkType + "] could not be found. Please check the documentation" +
                    " for a list of currently supported networks.");
        }
    }

    private static void tensorflowInstalled() throws IOException, InterruptedException {
        // TODO this might have to be changed to something like python -c 'import tensorflow' instead
        if(Runtime.getRuntime().exec("pip show tensorflow").waitFor() != 0 &&
                Runtime.getRuntime().exec("pip show tensorflow-gpu").waitFor() != 0){
            throw new IllegalArgumentException("Python tensorflow not installed on this machine. Please run 'pip install tensorflow[-gpu]' first.");
        }
    }

    private static String extractGenFiles(Path output) throws IOException {
        Files.createDirectory(Paths.get(output.toFile().getAbsolutePath(), "deepwater"));
        Files.createDirectory(Paths.get(output.toFile().getAbsolutePath(), "deepwater", "models"));

        for(Map.Entry<String, String> entry : GEN_FILES.entrySet()) {
            Files.copy(
                    TFPythonWrapper.class.getResource("/" + entry.getKey()).openStream(),
                    new File(output.toFile(), entry.getValue() + File.separator + entry.getKey()).toPath()
            );
        }

        return output.toFile().getAbsolutePath() + File.separator + GEN_SCRIPT;
    }

    private static String runGenScript(Path output,
                                       String runScript,
                                       String networkType,
                                       int width,
                                       int height,
                                       int channels,
                                       int numClasses,
                                       int[] hiddens) throws IOException, InterruptedException {
        // TODO set activation functions appropriately
        // TODO make a check in MLP if hidden != [200,200] gen the file
        // TODO delete files - but when?
        // TODO do we need to have the number of hidden layers in the filename?
        StringBuilder command = new StringBuilder("python "
                + runScript + " "
                + output.toFile().getAbsolutePath() + " "
                + networkType + " "
                + width + " "
                + height + " "
                + channels + " "
                + numClasses + " ");

        if(null != hiddens) {
            command.append("\"" + Arrays.toString(hiddens) + "\"");
        }

        Runtime.getRuntime().exec(command.toString()).waitFor();
        return output.toFile().getAbsolutePath() + File.separator +
                networkType + "_"
                + height + "x"
                + width + "x"
                + channels + "_"
                + numClasses
                + ".meta";
    }
}
