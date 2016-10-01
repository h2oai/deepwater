package deepwater.backends.tensorflow.models;


import deepwater.backends.BackendModel;

import static org.bytedeco.javacpp.tensorflow.*;


public class TensorflowModel implements BackendModel {

    private GraphDef graph;

    public TensorflowModel(GraphDef graph){
        this.graph = graph;
    }

    public GraphDef getGraph() {
        return graph;
    }
}
