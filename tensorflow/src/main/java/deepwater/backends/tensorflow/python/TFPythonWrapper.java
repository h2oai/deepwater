package deepwater.backends.tensorflow.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class TFPythonWrapper {
    static final String GEN_SCRIPT = "h2o_deepwater_generate_models.py";
    static final Set<String> GEN_FILES = new HashSet<String>() {{
        this.add(GEN_SCRIPT);
        this.add("deepwater/train.py");
        this.add("deepwater/optimizers.py");
        this.add("deepwater/models/alexnet.py");
        this.add("deepwater/models/inception.py");
        this.add("deepwater/models/lenet.py");
        this.add("deepwater/models/mlp.py");
        this.add("deepwater/models/nn.py");
        this.add("deepwater/models/resnet.py");
        this.add("deepwater/models/utils.py");
        this.add("deepwater/models/vgg.py");
        this.add("deepwater/models/__init__.py");
    }};

    public static String generateMetaFile(
            String networkType,
            int width,
            int height,
            int channels,
            int numClasses,
            int[] hiddens) {

        networkType = networkType.toLowerCase();
        if (networkType.endsWith("_bn")) {
            networkType = networkType.replace("_bn", "");
        }

        templateFilePresent(networkType);

        try {
            tensorflowInstalled();

            Path output = Files.createTempDirectory(Paths.get(System.getProperty("java.io.tmpdir")), Long.toString(System.nanoTime()));
            output.toFile().deleteOnExit();

            String runScript = extractGenFiles(output);

            return runGenScript(output, runScript, networkType, width, height, channels, numClasses, hiddens);
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException(e);
        }
    }

    static void templateFilePresent(String networkType) {
        if (null == TFPythonWrapper.class.getResource("/deepwater/models/" + networkType + ".py")) {
            throw new IllegalArgumentException("Neither a Tensorflow meta graph file nor a Python template file for network [" + networkType + "] could be " +
                    "found. If you are running a user defined network please make sure the path is correct. Otherwise please check the " +
                    "documentation for a list of currently supported networks.");
        }
    }

    static void tensorflowInstalled() throws IOException, InterruptedException {
        if (Runtime.getRuntime().exec(new String[]{"python", "-c", "import tensorflow"}).waitFor() != 0) {
            throw new IllegalArgumentException("Python Tensorflow not installed on this machine. Please run 'pip install tensorflow[-gpu]' first.");
        }
    }

    static String extractGenFiles(Path output) throws IOException {
        Files.createDirectory(Paths.get(output.toFile().getAbsolutePath(), "deepwater"));
        Files.createDirectory(Paths.get(output.toFile().getAbsolutePath(), "deepwater", "models"));

        for (String entry : GEN_FILES) {
            Files.copy(
                    TFPythonWrapper.class.getResource("/" + entry).openStream(),
                    new File(output.toFile(), entry).toPath()
            );
        }

        return output.toFile().getAbsolutePath() + File.separator + GEN_SCRIPT;
    }

    static String runGenScript(Path output,
                               String runScript,
                               String networkType,
                               int width,
                               int height,
                               int channels,
                               int numClasses,
                               int[] hiddens) throws IOException, InterruptedException {
        List<String> command = new ArrayList<String>() {{
            this.add("python");
            this.add(runScript);
            this.add(output.toFile().getAbsolutePath());
            this.add(networkType);
            this.add(width + "");
            this.add(height + "");
            this.add(channels + "");
            this.add(numClasses + "");
        }};

        if (null != hiddens) {
            command.add(Arrays.toString(hiddens));
        }

        String[] cmdArray = new String[command.size()];
        command.toArray(cmdArray);

        Runtime.getRuntime().exec(cmdArray).waitFor();
        return output.toFile().getAbsolutePath() + File.separator
                + networkType + "_"
                + width + "x"
                + height + "x"
                + channels + "_"
                + numClasses
                + ".meta";
    }
}
