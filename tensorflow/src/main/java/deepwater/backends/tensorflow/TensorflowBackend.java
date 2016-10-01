package deepwater.backends.tensorflow;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.datasets.ImageDataSet;


public class TensorflowBackend implements BackendTrain {

    @Override
    public void delete(BackendModel m) {

    }

    @Override
    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts, BackendParams backend_params, int num_classes, String name) {
        return ModelFactory.LoadModel(name);
    }

    @Override
    public void saveModel(BackendModel m, String model_path) {

    }

    @Override
    public void loadParam(BackendModel m, String param_path) {

    }

    @Override
    public void saveParam(BackendModel m, String param_path) {

    }

    @Override
    public String toJson(BackendModel m) {
        return null;
    }

    @Override
    public void setParameter(BackendModel m, String name, float value) {

    }

    @Override
    public float[] train(BackendModel m, float[] data, float[] label) {
        return new float[0];
    }

    @Override
    public float[] predict(BackendModel m, float[] data, float[] label) {
        return new float[0];
    }

    @Override
    public float[] predict(BackendModel m, float[] data) {
        return new float[0];
    }
}
