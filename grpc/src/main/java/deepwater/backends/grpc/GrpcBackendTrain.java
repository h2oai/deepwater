package deepwater.backends.grpc;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.datasets.ImageDataSet;

public class GrpcBackendTrain implements BackendTrain {

    public void delete(BackendModel m) {

    }

    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts, BackendParams backend_params, int num_classes, String name) {
        return null;
    }

    public void saveModel(BackendModel m, String model_path) {

    }

    public void loadParam(BackendModel m, String param_path) {

    }

    public void saveParam(BackendModel m, String param_path) {

    }

    public float[] loadMeanImage(BackendModel m, String path) {
        return new float[0];
    }

    public String toJson(BackendModel m) {
        return null;
    }

    public void setParameter(BackendModel m, String name, float value) {

    }

    public float[] train(BackendModel m, float[] data, float[] label) {
        return new float[0];
    }

    public float[] predict(BackendModel m, float[] data, float[] label) {
        return new float[0];
    }

    public float[] predict(BackendModel m, float[] data) {
        return new float[0];
    }
}
