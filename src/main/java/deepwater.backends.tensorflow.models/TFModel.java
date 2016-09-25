package deepwater.backends.tensorflow.models;


import static org.bytedeco.javacpp.tensorflow.*;


/**
 * Created by fmilo on 9/23/16.
 */
public class TFModel {

    private GraphDef graph;

    public TFModel(GraphDef graph){
        this.graph = graph;
    }

    public GraphDef getGraph() {
        return graph;
    }
}
