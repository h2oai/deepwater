package deepwater.backends;

import deepwater.datasets.ImageDataSet;

public interface BackendTrain {

  void delete(BackendModel m);

  // The method to construct a trainable Deep Water model.
  // name specifies the model architecture, or is a path to a graph definition file
  BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts,
      BackendParams backend_params, int num_classes, String name);

  void saveModel(BackendModel m, String model_path);

  void loadParam(BackendModel m, String param_path);

  void saveParam(BackendModel m, String param_path);

  /**
   * Computes the hash of model parameters file stored as byte[]. Backend specific as it might be the param file or a zipped archive
   * Mainly used for testing purpose.
   * @param param file containing all the parameters
   * @return
   */
  int paramHash(byte[] param);

  float[] loadMeanImage(BackendModel m, String path);

  String toJson(BackendModel m);

  // learning_rate
  // weight_decay
  // momentum
  // clip_gradient: bool
  void setParameter(BackendModel m, String name, float value);

  // data[mini_batch_size * input_neurons]
  // label[mini_batch_size]
  // return value is to be ignored
  // TODO: return void
  float[] train(BackendModel m, float[] data, float[] label);

  // data[mini_batch_size * input_neurons]
  // returns float[mini_batch_size * nclasses] with per-class probabilities (regression: nclasses=1)
  float[] predict(BackendModel m, float[] data);
}
