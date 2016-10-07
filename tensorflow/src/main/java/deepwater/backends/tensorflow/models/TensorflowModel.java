package deepwater.backends.tensorflow.models;


import deepwater.backends.BackendModel;
import org.bytedeco.javacpp.tensorflow;

import static org.bytedeco.javacpp.tensorflow.*;


public class TensorflowModel implements BackendModel {

    private GraphDef graph;

    protected Session session;

    public int frameSize;

    public TensorflowModel(GraphDef graph){
        this.graph = graph;
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
}
