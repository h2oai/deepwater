package deepwater.backends.tensorflow.models;


import com.google.common.io.ByteSink;
import deepwater.backends.BackendModel;
import deepwater.backends.tensorflow.TensorflowMetaModel;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class TensorflowModel implements BackendModel {

    public TensorflowMetaModel meta;
    public int classes;
    public int frameSize;
    protected Session session;
    private Graph graph;
    private Map<String, Float> parameters;
    private byte[] modelGraphData;
    public int miniBatchSize;
    public String[] activations = null;

    public static final Map<String, Integer> activationToNumeric = new HashMap<String, Integer>() {{
      this.put("relu", 0);
      this.put("tanh", 1);
    }};

    public float inputDropoutRatio;
    public float[] hiddenDropoutRatios;

    TensorflowModel(TensorflowMetaModel meta, Graph graph, byte[] definition) {
        this.meta = meta;
        this.graph = graph;
        this.parameters = new HashMap<>();
        modelGraphData = definition;
    }

    public Graph getGraph() {
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

    public float getParameter(String name, float defaultValue) {
        if (!parameters.containsKey(name)){
            return defaultValue;
        }
        return parameters.get(name);
    }

    public void saveModel(String path) throws IOException {
        ByteSink bs = com.google.common.io.Files.asByteSink(new File(path));
        bs.write(modelGraphData);
    }

    public FloatBuffer createDataMatrix(float[] data) {
        assert data.length == frameSize * miniBatchSize;
        FloatBuffer buffer = FloatBuffer.allocate(frameSize * miniBatchSize);
        buffer.put(data);
        return (FloatBuffer) buffer.flip();
    }

    public float[] getPredictions(Tensor tensor) {
        float[][] predictions = new float[miniBatchSize][classes];
        tensor.copyTo(predictions);
        float[] flatten = new float[miniBatchSize * classes];
        int start = 0;
        int length = classes;
        for (int i = 0; i < predictions.length; i++) {
            System.arraycopy(predictions[i], 0, flatten, start, length);
            start += classes;
        }
        return flatten;
    }

}
