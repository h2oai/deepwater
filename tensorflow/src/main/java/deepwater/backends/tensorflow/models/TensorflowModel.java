package deepwater.backends.tensorflow.models;


import com.google.common.io.ByteSink;
import com.google.common.io.ByteSource;
import com.google.common.io.ByteStreams;
import com.google.common.io.Resources;
import deepwater.backends.BackendModel;
import deepwater.backends.tensorflow.TensorflowMetaModel;
import org.bytedeco.javacpp.tensorflow;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

import static deepwater.backends.tensorflow.models.ModelFactory.checkStatus;
import static org.bytedeco.javacpp.tensorflow.Env;
import static org.bytedeco.javacpp.tensorflow.GraphDef;
import static org.bytedeco.javacpp.tensorflow.ReadBinaryProto;
import static org.bytedeco.javacpp.tensorflow.Session;


public class TensorflowModel implements BackendModel {

    public TensorflowMetaModel meta;
    public int classes;
    public int frameSize;
    protected Session session;
    private GraphDef graph;
    private Map<String, Float> parameters;
    private byte[] modelGraphData;

    TensorflowModel(TensorflowMetaModel meta, GraphDef graph, byte[] definition) {
        this.meta = meta;
        this.graph = graph;
        this.parameters = new HashMap<>();
        modelGraphData = definition;
    }

    public GraphDef getGraph() {
        return graph;
    }

    public Session getSession() {
        return session;
    }

    public void setSession(Session session) {
        this.session = session;
    }

    public void setParameter(String name, float value) {
        parameters.put(name, value);
    }

    public void saveModel(String path) throws IOException {
        ByteSink bs = com.google.common.io.Files.asByteSink(new File(path));
        bs.write(modelGraphData);
    }

    private tensorflow.GraphDef extractGraphDefinition(String resourceModelName) {
        tensorflow.GraphDef graph_def = new tensorflow.GraphDef();
        try {
            String path;
            URL url = Resources.getResource(resourceModelName);
            modelGraphData = Resources.toByteArray(url);
            path = saveToTempFile(modelGraphData);

            if (modelGraphData == null || modelGraphData.length == 0) {
                InputStream in = ModelFactory.class.getResourceAsStream("/" + resourceModelName);
                if (in != null) {
                    path = saveToTempFile(in);
                    modelGraphData = ByteStreams.toByteArray(in);
                } else {
                    // FIXME: for some reason inside idea it does not work
                    path = "/home/fmilo/workspace/deepwater/tensorflow/src/main/resources/" + resourceModelName;
                    ByteSource bs = com.google.common.io.Files.asByteSource( new File(path));
                    modelGraphData = bs.read();
                }
            }

            checkStatus(ReadBinaryProto(Env.Default(), path, graph_def));
        } catch (IOException e) {
            e.printStackTrace();
            //throw new InvalidArgumentException(new String[]{"could not load model " + model_name});
        }
        return graph_def;
    }

    private String saveToTempFile(byte[] in) throws IOException {
        String path;
        File temp = File.createTempFile("tempfile", ".tmp");
        ByteSink bs = com.google.common.io.Files.asByteSink(temp);
        bs.write(in);
        path = temp.getAbsolutePath();
        return path;
    }

    private String saveToTempFile(InputStream in) throws IOException {
        String path;
        File temp = File.createTempFile("tempfile", ".tmp");
        BufferedWriter bw = new BufferedWriter(new FileWriter(temp));
        bw.close();
        java.nio.file.Files.copy(in, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
        path = temp.getAbsolutePath();
        return path;
    }
}
