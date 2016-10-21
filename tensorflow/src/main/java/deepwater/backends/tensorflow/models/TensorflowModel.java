package deepwater.backends.tensorflow.models;


import deepwater.backends.BackendModel;
import deepwater.backends.tensorflow.TensorflowMetaModel;
import org.bytedeco.javacpp.tensorflow;

import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.*;


public class TensorflowModel implements BackendModel {

    private GraphDef graph;
    public TensorflowMetaModel meta;
    public int classes;

    protected Session session;

    public int frameSize;
    private Map<String, Float> parameters;

    public TensorflowModel(TensorflowMetaModel meta, GraphDef graph){
        this.meta = meta;
        this.graph = graph;
        this.parameters = new HashMap<>();
    }

    public GraphDef getGraph() {
        return graph;
    }

    public void setSession(Session session) {
        this.session = session;
    }

    public Session getSession() {
        return session;
    }

    public void setParameter(String name, float value) {
        parameters.put(name, value);
    }
}
