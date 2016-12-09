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

public class GrpcBackendTrain implements BackendTrain {

    // assumes the current class is called MyLogger
    private final static Logger log = Logger.getLogger(GrpcBackendTrain.class.getName());

    private final Client client;

    public GrpcBackendTrain(String host, int port){
        client = new Client(host, port);
    }

    public void delete(BackendModel m) {
        try {
            client.deleteModel(m);
        } catch (Exception e) {
            log.log(Level.WARNING, e.getMessage());
        }

    }

    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts, BackendParams backend_params, int num_classes, String name) {
        return null;
    }

    public void saveModel(BackendModel m, String model_path) {
        try {
            client.saveModel(m, model_path);
        } catch (Exception e) {
            log.log(Level.WARNING, e.getMessage());
        }
    }

    public void loadParam(BackendModel m, String param_path) {
        try {
            client.loadWeights(m, param_path);
        } catch (Exception e) {
            log.log(Level.WARNING, e.getMessage());
        }

    }

    public void saveParam(BackendModel m, String param_path) {
        try {
            client.saveWeights(m, param_path);
        } catch (Exception e) {
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

    public void setParameter(BackendModel m, String name, float value) {
        try {
            HashMap<String, Object> params = new HashMap<>();
            params.put(name, value);
            client.setParameters(m, params);
        } catch (Exception e) {
            log.log(Level.WARNING, e.getMessage());
        }

    }

    public float[] train(BackendModel m, float[] data, float[] label) {
        try {
            HashMap<String, float[]> params = new HashMap<>();
            params.put("batch_image_data", data);
            params.put("categorical_labels", label);
            String[] fetches = new String[]{"categorical_logits"};
            Map<String, float[]> results = client.train(m, params, fetches);
            return results.get("categorical_logits");
        } catch (Exception e) {
            log.log(Level.WARNING, e.getMessage());
            return new float[]{};
        }
    }

    public float[] predict(BackendModel model, float[] data, float[] label) {
        try {
            HashMap<String, float[]> m = new HashMap<>();
            m.put("batch_image_data", data);
            m.put("categorical_labels", label);
            String[] fetches = new String[]{"categorical_logits"};
            Map<String, float[]> results = client.predict(model, m, fetches);
            return results.get("categorical_logits");
        } catch (Exception e) {
            log.log(Level.WARNING, e.getMessage());
            return new float[]{};
        }
    }

    public float[] predict(BackendModel model, float[] data) {
        try {
            HashMap<String, float[]> m = new HashMap<>();
            m.put("batch_image_data", data);
            String[] fetches = new String[]{"categorical_logits"};
            Map<String, float[]> results = client.predict(model, m, fetches);
            return results.get(fetches[0]);
        } catch (Exception e) {
            log.log(Level.WARNING, e.getMessage());
            return new float[]{};
        }
    }
}
