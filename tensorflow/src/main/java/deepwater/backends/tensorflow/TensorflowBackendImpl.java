package deepwater.backends.tensorflow;


import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TensorflowModel;

import java.io.FileNotFoundException;
import java.io.InputStream;

public class TensorflowBackendImpl {

    public void delete() {

    }

    public void createNetwork(String name) {

        try {
            TensorflowModel model = ModelFactory.LoadModel("cifarnet");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

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
