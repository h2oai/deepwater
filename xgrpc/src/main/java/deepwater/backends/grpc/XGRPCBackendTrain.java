package deepwater.backends.grpc;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.datasets.ImageDataSet;

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import deepwater.utils.PythonWorkerPool;

public class XGRPCBackendTrain implements BackendTrain {

    // assumes the current class is called MyLogger
    private final static Logger log = Logger.getLogger(XGRPCBackendTrain.class.getName());

    private final Client client;

    private static PythonWorkerPool pypool = new PythonWorkerPool();

    public XGRPCBackendTrain() {
        HashMap<String, String> env = new HashMap<>();
        String userHome = System.getProperty("user.home");
        env.put("PYTHONPATH", userHome + "/deepwater/xgrpc/src/main/python/");
        env.put("LD_LIBRARY_PATH", "/usr/local/cuda/lib64");
        pypool.createPythonWorker(userHome + "/anaconda3/envs/deepwater/bin/python", env);
        client = new Client("localhost", 50051);
    }
    public XGRPCBackendTrain(String host, int port){
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

    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions runtimeOptions, BackendParams backend_params, int num_classes, String name) {
        try {

            XGRPCBackendSession session = client.createSession(runtimeOptions);

            XGRPCBackendModel model = client.createModel(session, name, backend_params.asMap());
            model.setSession(session);

            return model;
        } catch (Exception e) {
            e.printStackTrace();
            log.log(Level.WARNING, e.getMessage());
            return null;
        }
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
            client.loadWeights(model.getSession(), model, param_path);
        } catch (Exception e) {
            e.printStackTrace();
            log.log(Level.WARNING, e.getMessage());
        }

    }

    public void saveParam(BackendModel m, String param_path) {
        try {
            XGRPCBackendModel model = (XGRPCBackendModel) m;
            client.saveWeights(model.getSession(), model, param_path);
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
             int batchSize =  10;
             int classes = 10;

                float[] labelData = new float[Math.toIntExact(batchSize * classes)];
                for (int i = 0; i < batchSize; i++) {
                    int idx = (int)label[i];
                    labelData[i * classes + idx] = (float) 1.0;
                }


            HashMap<String, float[]> params = new HashMap<>();
            params.put("batch_image_input", data);
            params.put("categorical_labels", labelData);
            String[] fetches = new String[]{"categorical_logits"};
            XGRPCBackendModel _model = (XGRPCBackendModel) model;
            Map<String, float[]> results = client.train(_model.getSession(), _model, params, fetches);
            return results.get("categorical_logits");
        } catch (Exception e) {
            e.printStackTrace();
            log.log(Level.WARNING, e.getMessage());
            return new float[]{};
        }
    }

    public float[] predict(BackendModel model, float[] data, float[] label) {
        try {
            int batchSize =  10;
            int classes = 10;

            float[] labelData = new float[Math.toIntExact(batchSize * classes)];
            for (int i = 0; i < batchSize; i++) {
                int idx = (int)label[i];
                labelData[i * classes + idx] = (float) 1.0;
            }

            HashMap<String, float[]> m = new HashMap<>();
            m.put("batch_image_input", data);
            m.put("categorical_labels", labelData);
            String[] fetches = new String[]{"categorical_logits"};
            XGRPCBackendModel _model = (XGRPCBackendModel) model;
            Map<String, float[]> results = client.predict(_model.getSession(), _model, m, fetches);
            return results.get("categorical_logits");
        } catch (Exception e) {
            e.printStackTrace();
            log.log(Level.WARNING, e.getMessage());
            return new float[]{};
        }
    }

    public float[] predict(BackendModel model, float[] data) {
        try {
            HashMap<String, float[]> m = new HashMap<>();
            m.put("batch_image_input", data);
            String[] fetches = new String[]{"categorical_logits"};
            XGRPCBackendModel _model = (XGRPCBackendModel) model;
            Map<String, float[]> results = client.predict(_model.getSession(), _model, m, fetches);
            return results.get(fetches[0]);
        } catch (Exception e) {
            e.printStackTrace();
            log.log(Level.WARNING, e.getMessage());
            return new float[]{};
        }
    }
}
