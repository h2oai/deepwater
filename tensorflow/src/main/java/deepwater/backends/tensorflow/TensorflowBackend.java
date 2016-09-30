package deepwater.backends.tensorflow;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.datasets.ImageDataSet;

public class TensorflowBackend extends TensorflowBackendImpl implements BackendTrain {

    @Override
    public void delete(BackendModel m) {

    }

    @Override
    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts, BackendParams backendParams, int num_classes, String name) {
        assert(opts!=null);
        assert(dataset!=null);
        assert(backendParams !=null);

        // TensorflowBackendModel m = super.createNetwork(name, num_classes);

        if (backendParams.get("clip_gradient") != null)
            //_mxnet.setClipGradient(((Double)bparms.get("clip_gradient")).floatValue());
        if (backendParams.get("hidden") == null) {
            //getTrainer().buildNet(num_classes, ((Integer) bparms.get("mini_batch_size")).intValue(), name);
        } else {
//            getTrainer().buildNet(
//                    num_classes,
//                    ((Integer) backendParams.get("mini_batch_size")).intValue(),
//                    name,
//                    ((int[]) backendParams.get("hidden")).length,
//                    (int[]) backendParams.get("hidden"),
//                    (String[]) backendParams.get("activations"),
//                    ((Double) backendParams.get("input_dropout_ratio")).doubleValue(),
//                    (double[]) backendParams.get("hidden_dropout_ratios")
//            );
        }
        return new TensorflowBackendModel();
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
