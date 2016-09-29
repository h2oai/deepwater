package deepwater.backends.tensorflow;


import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TFModel;

import java.io.InputStream;

public class TensorflowBackendImpl {

    public void delete() {

    }

    public void createNetwork(String name) {

        InputStream stream = getClass().getResourceAsStream("mnist.pb");

        TFModel model = ModelFactory.LoadModel("LENET");

    }

    public void saveModel(String model_path) {

    }

    public void loadParam(String param_path) {

    }

    public void saveParam(String param_path) {

    }

    public String toJson() {
        return null;
    }

    public void setParameter(String name, float value) {

    }

    public float[] train(float[] data, float[] label) {
        return new float[0];
    }

    public float[] predict(float[] data, float[] label) {
        return new float[0];
    }

    public float[] predict(float[] data) {
        return new float[0];
    }

}
