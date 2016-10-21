package deepwater.backends;
import java.io.*;

import deepwater.datasets.ImageDataSet;

public interface BackendTrain {

    void delete(BackendModel m);

    BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts,
                          BackendParams backend_params, int num_classes, String name);

    void saveModel(BackendModel m, String model_path);

    void loadParam(BackendModel m, String param_path);

    void saveParam(BackendModel m, String param_path);

    float[] loadMeanImage(BackendModel m, String path);

    String toJson(BackendModel m);

    // learning_rate
    // weight_decay
    // momentum
    // clip_gradient: bool
    void setParameter(BackendModel m, String name, float value);

float[] train(BackendModel m, float[] data, float[] label);

    float[] predict(BackendModel m, float[] data, float[] label);

    float[] predict(BackendModel m, float[] data);
}
