package deepwater.backends.tensorflow.models;

import com.google.gson.Gson;
import deepwater.backends.tensorflow.TensorflowMetaModel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
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

        if (src != null) {
            URL jar = src.getLocation();
            ZipInputStream zip = new ZipInputStream(jar.openStream());
            ZipEntry ze = null;

            while ((ze = zip.getNextEntry()) != null) {
                String entryName = ze.getName();
                list.add(entryName);
                System.out.println(list);
            }

        }

    }

    static public TensorflowModel LoadModel(String modelName) throws FileNotFoundException {
        String resourceModelName = convertToCanonicalName(modelName);
        String resourceMetaModelName = convertToCanonicalMetaName(modelName);
        GraphDef graph_def = extractGraphDefinition(resourceModelName);
        TensorflowMetaModel meta = extractMetaModel(resourceMetaModelName);
        return new TensorflowModel(meta, graph_def);
    }

    private static TensorflowMetaModel extractMetaModel(String resourceMetaModelName) throws FileNotFoundException {
        Gson g = new Gson();
        InputStream in = ModelFactory.class.getResourceAsStream(resourceMetaModelName);
        if (in == null){
            String path = "/home/fmilo/workspace/deepwater/tensorflow/src/main/resources/" + resourceMetaModelName;
            in =  new FileInputStream(path);
        }
        Reader reader = null;
        try {
            reader = new InputStreamReader(in, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        return g.fromJson(reader, TensorflowMetaModel.class);
    }

    private static GraphDef extractGraphDefinition(String resourceModelName) {
        GraphDef graph_def = new GraphDef();
        try {
            //debugJar();
            String path;
            InputStream in = ModelFactory.class.getResourceAsStream(resourceModelName);
            if (in != null) {
                path = saveToTempFile(in);
            } else {
                // FIXME: for some reason inside idea it does not work
                path = "/home/fmilo/workspace/deepwater/tensorflow/src/main/resources/" + resourceModelName;
            }
            checkStatus(ReadBinaryProto(Env.Default(), path, graph_def));
        } catch (IOException e) {
            e.printStackTrace();
            //throw new InvalidArgumentException(new String[]{"could not load model " + model_name});
        }
        return graph_def;
    }

    private static String saveToTempFile(InputStream in) throws IOException {
        String path;
        File temp = File.createTempFile("tempfile", ".tmp");
        BufferedWriter bw = new BufferedWriter(new FileWriter(temp));
        bw.close();
        Files.copy(in, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
        path = temp.getAbsolutePath();
        return path;
    }

    private static String convertToCanonicalMetaName(String model_name) {
        return model_name.toLowerCase() + ".meta";
    }

    private static String convertToCanonicalName(String model_name) {
        return model_name.toLowerCase() + ".pb";
    }

    static void checkStatus(Status status) {
        if (!status.ok()) {
            throw new InternalError(status.error_message().getString());
        }
    }
}
