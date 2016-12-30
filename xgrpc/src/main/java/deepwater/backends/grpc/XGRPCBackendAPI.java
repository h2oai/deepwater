package deepwater.backends.grpc;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendAPI;
import deepwater.backends.RuntimeOptions;
import deepwater.datasets.ImageDataSet;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import deepwater.datasets.Pair;
import deepwater.utils.PythonWorkerPool;

public class XGRPCBackendAPI implements BackendAPI {

  // assumes the current class is called MyLogger
  private static final Logger log = Logger.getLogger(XGRPCBackendAPI.class.getName());

  private final Client client;

  private static PythonWorkerPool pypool = new PythonWorkerPool();

  public XGRPCBackendAPI() {
    HashMap<String, String> env = new HashMap<>();
    String userHome = System.getProperty("user.home");
    env.put("PYTHONPATH", userHome + "/deepwater/xgrpc/src/main/python/");
    env.put("LD_LIBRARY_PATH", "/usr/local/cuda/lib64");
    pypool.createPythonWorker(userHome + "/anaconda3/envs/deepwater/bin/python", env);
    client = new Client("localhost", 50051);
  }

  public XGRPCBackendAPI(String host, int port) {
    client = new Client(host, port);
  }

  public void delete(BackendModel m) {
    try {
      XGRPCBackendModel model = (XGRPCBackendModel) m;
      client.deleteSession(model.getSession(), m);
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
    }
  }

  public BackendModel buildNet(
      ImageDataSet dataset,
      RuntimeOptions runtimeOptions,
      BackendParams backend_params,
      int num_classes,
      String name) {
    try {
      XGRPCBackendSession session = client.createSession(runtimeOptions);
      if (new File(name).exists()) {
        Map<String, Object> params = new HashMap<>();
        params.put("path", name);
        XGRPCBackendModel model = client.loadModel(session, name);
        model.setSession(session);
      } else {
        XGRPCBackendModel model = client.createModel(session, name, backend_params.asMap());
        model.setDataset(dataset);
        model.setSession(session);
        return model;
      }
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
      return null;
    }
    return null;
  }

  public void saveModel(BackendModel m, String model_path) {
    try {
      XGRPCBackendModel model = (XGRPCBackendModel) m;
      client.saveModel(model.getSession(), model, model_path);
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
    }
  }

  public void loadParam(BackendModel m, String param_path) {
    try {
      XGRPCBackendModel model = (XGRPCBackendModel) m;
      client.loadModelVariables(model.getSession(), model, param_path);
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
    }
  }

  public void saveParam(BackendModel m, String param_path) {
    try {
      XGRPCBackendModel model = (XGRPCBackendModel) m;
      client.saveModelVariables(model.getSession(), model, param_path);
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
    }
  }

  @Deprecated
  public float[] loadMeanImage(BackendModel m, String path) {
    log.log(Level.WARNING, "DEPRECATED");
    return new float[0];
  }

  @Deprecated
  public String toJson(BackendModel m) {
    log.log(Level.WARNING, "DEPRECATED");
    return null;
  }

  public void setParameter(BackendModel model, String name, float value) {
    try {
      HashMap<String, Object> params = new HashMap<>();
      params.put(name, value);
      XGRPCBackendModel _model = (XGRPCBackendModel) model;
      client.setParameters(_model.getSession(), _model, params);
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
    }
  }

  public float[] train(BackendModel model, float[] data, float[] label) {
    try {
      XGRPCBackendModel _model = (XGRPCBackendModel) model;

      ImageDataSet backend = _model.getBackend();

      int classes = backend.getNumClasses();
      int batchSize = label.length;

      float[] labelData = new float[Math.toIntExact(batchSize * classes)];
      for (int i = 0; i < batchSize; i++) {
        int idx = (int) label[i];
        labelData[i * classes + idx] = (float) 1.0;
      }

      HashMap<String, Pair<float[], int[]>> params = new HashMap<>();
      params.put("batch_image_input", new Pair<>(data, new int[]{batchSize, data.length/batchSize}));
      params.put("categorical_labels", new Pair<>(labelData, new int[]{batchSize, classes}));

      String[] fetches = new String[] {"categorical_logits", "train"};
      Map<String, float[]> results = client.train(_model.getSession(), _model, params, fetches);
      return results.get("categorical_logits");
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
      return new float[] {};
    }
  }

  public float[] predict(BackendModel model, float[] data, float[] label) {
    try {
      XGRPCBackendModel _model = (XGRPCBackendModel) model;
      int classes = label.length;
      int batchSize = label.length;

      if (_model.getBackend() != null) {
        classes = _model.getBackend().getNumClasses();
        batchSize = label.length;
      }

      float[] labelData = new float[Math.toIntExact(batchSize * classes)];
      for (int i = 0; i < batchSize; i++) {
        int idx = (int) label[i];
        labelData[i * classes + idx] = (float) 1.0;
      }

      HashMap<String, Pair<float[], int[]>> m = new HashMap<>();
      m.put("batch_image_input", new Pair<>(data, new int[]{batchSize, data.length/batchSize}));
      m.put("categorical_labels", new Pair<>(labelData, new int[]{batchSize, classes}));

      String[] fetches = new String[] {"categorical_logits", "total_loss"};
      Map<String, float[]> results = client.predict(_model.getSession(), _model, m, fetches);
      printAverage("total_loss", results.get("total_loss"));
      return results.get("categorical_logits");
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
      return new float[] {};
    }
  }

  private double printAverage(String name, float[] loss) {
    System.out.println(name);
    double average = 0.0;
    double sum = 0.0;
    for (float l : loss) {
      sum += l;
    }
    average = sum / (loss.length * 1.0);
    System.out.print(average);
    System.out.println();
    return average;
  }

  public float[] predict(BackendModel model, float[] data) {
    try {
      XGRPCBackendModel _model = (XGRPCBackendModel) model;
      int classes = data.length;
      int batchSize = data.length;

      if (_model.getBackend() != null) {
        classes = _model.getBackend().getNumClasses();
        batchSize = data.length;
      }
      //FIXME: the size of the dataset should be explicitly set here.
      int imageSize = b.getWidth() * b.getHeight() * b.getChannels();

      HashMap<String, Pair<float[], int[]>> m = new HashMap<>();
      m.put("batch_image_input", new Pair<>(data, new int[]{data.length/imageSize, imageSize}));

      String[] fetches = new String[] {"categorical_logits"};

      Map<String, float[]> results = client.predict(_model.getSession(), _model, m, fetches);
      return results.get(fetches[0]);
    } catch (Exception e) {
      e.printStackTrace();
      log.log(Level.WARNING, e.getMessage());
      return new float[] {};
    }
  }
}
