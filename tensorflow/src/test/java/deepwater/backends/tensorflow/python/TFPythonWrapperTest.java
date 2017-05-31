package deepwater.backends.tensorflow.python;

import org.junit.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.*;

public class TFPythonWrapperTest {

    @Test
    public void shouldExtractFilesFromJar() throws Exception {
        Path output = Files.createTempDirectory(Paths.get(System.getProperty("java.io.tmpdir")), Long.toString(System.nanoTime()));
        output.toFile().deleteOnExit();

        TFPythonWrapper.extractGenFiles(output);

        for (String entry : TFPythonWrapper.GEN_FILES) {
            assertTrue(new File(output.toFile().getAbsoluteFile() + "/" + entry).exists());
        }
    }

    @Test
    public void shouldFindSupportedNetworkTemplates() throws Exception {
        Set<String> supportedNetworks = new HashSet<>();
        supportedNetworks.add("lenet");
        supportedNetworks.add("mlp");
        supportedNetworks.add("inception");
        supportedNetworks.add("resnet");
        supportedNetworks.add("vgg");
        supportedNetworks.add("alexnet");
        for (String supportedNetwork : supportedNetworks) {
            try {
                TFPythonWrapper.templateFilePresent(supportedNetwork);
            } catch (Exception e) {
                fail("Template file not found for " + supportedNetwork);
            }
        }

        try {
            TFPythonWrapper.templateFilePresent("NONEXISTINGNETWORK");
        } catch (Exception e) {
            return;
        }
        fail("Found template for non existing network NONEXISTINGNETWORK");
    }

}