package deepwater.backends.caffe;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.datasets.ImageDataSet;

public class CaffeBackend implements BackendTrain {
  @Override
  public void delete(BackendModel m) {
  }

  @Override
  public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts, BackendParams backend_params, int num_classes, String name) {
    return new BackendModel() {
      @Override
      public int hashCode() {
        return super.hashCode();
      }
    };
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
