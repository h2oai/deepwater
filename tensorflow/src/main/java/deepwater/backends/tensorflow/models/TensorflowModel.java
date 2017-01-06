package deepwater.backends.tensorflow.models;


import com.google.common.io.ByteSink;
import deepwater.backends.BackendModel;
import deepwater.backends.tensorflow.TensorflowMetaModel;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.GraphDef;
import static org.bytedeco.javacpp.tensorflow.Session;


public class TensorflowModel implements BackendModel {

    public TensorflowMetaModel meta;
    public int classes;
    public int frameSize;
    protected Session session;
    private GraphDef graph;
    private Map<String, Float> parameters;
    private byte[] modelGraphData;
    public int miniBatchSize;
    public String[] activations;
    public double inputDropoutRatio;
    public double[] hiddenDropoutRatios;

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

    public float getParameter(String name) {
        return parameters.get(name);
    }

    public void saveModel(String path) throws IOException {
        ByteSink bs = com.google.common.io.Files.asByteSink(new File(path));
        bs.write(modelGraphData);
    }

}
