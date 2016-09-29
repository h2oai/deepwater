package deepwater.backends.tensorflow.models;

import java.io.IOException;
import java.net.URL;
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

        GraphDef graph_def = new GraphDef();
//        try {
//            File temp = null;
//            debugJar();
//            InputStream in = ModelFactory.class.getResourceAsStream("mnist.pb");
//            temp = File.createTempFile("tempfile", ".tmp");
//            BufferedWriter bw = new BufferedWriter(new FileWriter(temp));
//            bw.close();
//            Files.copy(in, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
//        } catch (IOException e) {
//            e.printStackTrace();
//            //throw new InvalidArgumentException(new String[]{"could not load model " + model_name});
//        }

        String path = "src/main/resources/deepwater.backends.tensorflow.models/mnist.pb";
        checkStatus(ReadBinaryProto(Env.Default(), path, graph_def));
        return new TFModel(graph_def);

    }

    static void checkStatus(Status status) {
        if (!status.ok()) {
            throw new InternalError(status.error_message().getString());
        }
    }
}
