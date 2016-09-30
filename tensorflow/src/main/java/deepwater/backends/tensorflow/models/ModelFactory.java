package deepwater.backends.tensorflow.models;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.CodeSource;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import static org.bytedeco.javacpp.tensorflow.*;

public class ModelFactory {

    static public void debugJar() throws IOException {
        CodeSource src = ModelFactory.class.getProtectionDomain().getCodeSource();
        List<String> list = new ArrayList<>();

        if( src != null ) {
            URL jar = src.getLocation();
            ZipInputStream zip = new ZipInputStream( jar.openStream());
            ZipEntry ze = null;

            while( ( ze = zip.getNextEntry() ) != null ) {
                String entryName = ze.getName();
                list.add( entryName  );
                System.out.println(list) ;
            }

        }

    }

    static public TFModel LoadModel(String model_name) {
        String resource_model_name = convertToCanonicalName(model_name);
        GraphDef graph_def = new GraphDef();
        try {
            //debugJar();
            String path;
            InputStream in = ModelFactory.class.getResourceAsStream(resource_model_name);
            if (in != null) {
                File temp = File.createTempFile("tempfile", ".tmp");
                BufferedWriter bw = new BufferedWriter(new FileWriter(temp));
                bw.close();
                Files.copy(in, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
                path = temp.getAbsolutePath();
            } else {
                path = model_name;
            }
            checkStatus(ReadBinaryProto(Env.Default(), path, graph_def));
        } catch (IOException e) {
            e.printStackTrace();
            //throw new InvalidArgumentException(new String[]{"could not load model " + model_name});
        }

        return new TFModel(graph_def);

    }

    private static String convertToCanonicalName(String model_name) {
        return model_name.toLowerCase()  + ".pb";
    }

    static void checkStatus(Status status) {
        if (!status.ok()) {
            throw new InternalError(status.error_message().getString());
        }
    }
}
