package deepwater.backends.mxnet;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.datasets.ImageDataSet;

import java.io.*;

public class MXNetBackend implements BackendTrain {

  static private class MXNetLoader {
    static { //only load libraries once
      try {
        final boolean GPU = System.getenv("CUDA_PATH") != null;
        if (GPU) {
          System.out.println("Found CUDA_PATH environment variable, trying to connect to GPU devices.");
          //System.out.println(logNvidiaStats());
          System.out.println("Loading CUDA library.");
          util.loadCudaLib();
        } else {
          System.out.println("No GPU found - not loading CUDA library.");
        }
        System.out.println("Loading mxnet library.");
        util.loadNativeLib("mxnet");
        System.out.println("Loading H2O mxnet bindings.");
        util.loadNativeLib("Native");
      } catch (IOException e) {
        e.printStackTrace();
        throw new IllegalArgumentException("Couldn't load native libraries");
      }
    }
  }

  MXNetBackendModel get(BackendModel m) {
    return (MXNetBackendModel) m;
  }

  @Override
  public void delete(BackendModel m) {
    get(m).delete();
  }

  @Override
  public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts, BackendParams bparms, int num_classes, String name) {
    System.out.println("Loading H2O mxnet bindings.");
    new MXNetLoader();
    System.out.println("Done loading H2O mxnet bindings.");
    assert(opts!=null);
    assert(dataset!=null);
    assert(bparms!=null);
    System.out.println("Constructing model.");
    MXNetBackendModel mxnet = new MXNetBackendModel(dataset.getWidth(), dataset.getHeight(), dataset.getChannels(),
            opts.getDeviceID()[0], (int)opts.getSeed(), opts.useGPU());
    System.out.println("Done constructing model.");

    if (bparms.get("clip_gradient") != null)
      mxnet.setClipGradient(((Double)bparms.get("clip_gradient")).floatValue());
    System.out.println("Building network.");
    if (bparms.get("hidden") == null) {
      mxnet.buildNet(num_classes, ((Integer) bparms.get("mini_batch_size")).intValue(), name);
    } else {
      mxnet.buildNet(
              num_classes,
              ((Integer) bparms.get("mini_batch_size")).intValue(),
              name,
              ((int[]) bparms.get("hidden")).length,
              (int[]) bparms.get("hidden"),
              (String[]) bparms.get("activations"),
              ((Double) bparms.get("input_dropout_ratio")).doubleValue(),
              (double[]) bparms.get("hidden_dropout_ratios")
      );
    }
    System.out.println("Done building network.");
    return mxnet;
  }


  @Override
  public void setParameter(BackendModel m, String name, float value) {
    MXNetBackendModel mxnet = get(m);
    if (name == "momentum") {
      mxnet.setMomentum(value);
    } else if (name == "learning_rate") {
      mxnet.setLR(value);
    } else if (name == "clip_gradient") {
      mxnet.setClipGradient(value);
    } else throw new IllegalArgumentException("invalid parameter: "+name);
  }

  @Override
  public float[] train(BackendModel m, float[] data, float[] label) {
    get(m).train(data, label);
    return null;
  }

  @Override
  public float[] predict(BackendModel m, float[] data) {
    return get(m).predict(data);
  }

  @Override
  public void loadParam(BackendModel m, String networkParms) {
    MXNetBackendModel model = get(m);

    if (networkParms != null && !networkParms.isEmpty()) {
      File f = new File(networkParms);
      if (!f.exists() || f.isDirectory()) {
        //Log.err("Parameter file " + f + " not found.");
      } else {
        //Log.info("Loading the parameters (weights/biases) from: " + f.getAbsolutePath());
        model.loadParam(f.getAbsolutePath());
      }
    } else {
      //Log.warn("No network parameters file specified. Starting from scratch.");
    }
  }

  @Override
  public void writeParams(File file, byte[] payload) throws IOException {
    FileOutputStream os = new FileOutputStream(file.toString());
    os.write(payload);
    os.close();
  }

  @Override
  public void saveModel(BackendModel m, String model_path) {
    get(m).saveModel(model_path);
  }

  @Override
  public void saveParam(BackendModel m, String param_path) {
    get(m).saveParam(param_path);
  }

  @Override
  public byte[] readParams(File file) throws IOException {
    FileInputStream is = new FileInputStream(file);
    byte[] params = new byte[(int)file.length()];
    is.read(params);
    is.close();
    return params;
  }

  @Override
  public float[] loadMeanImage(BackendModel m, String param_path) {
    return get(m).loadMeanImage(param_path);
  }

  @Override
  public String toJson(BackendModel m) {
    return null;
  }

}
