package deepwater.backends.tensorflow;

import com.google.common.primitives.Floats;
import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TensorflowModel;
import deepwater.datasets.ImageDataSet;
import org.bytedeco.javacpp.tensorflow;
import org.tensorflow.framework.MetaGraphDef;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.DT_FLOAT;
import static org.bytedeco.javacpp.tensorflow.DT_INT32;
import static org.bytedeco.javacpp.tensorflow.DT_INT64;
import static org.bytedeco.javacpp.tensorflow.DT_STRING;
import static org.bytedeco.javacpp.tensorflow.Session;
import static org.bytedeco.javacpp.tensorflow.SessionOptions;
import static org.bytedeco.javacpp.tensorflow.Status;
import static org.bytedeco.javacpp.tensorflow.StringArray;
import static org.bytedeco.javacpp.tensorflow.StringTensorPairVector;
import static org.bytedeco.javacpp.tensorflow.StringVector;
import static org.bytedeco.javacpp.tensorflow.Tensor;
import static org.bytedeco.javacpp.tensorflow.TensorShape;
import static org.bytedeco.javacpp.tensorflow.TensorVector;


public class TensorflowBackend implements BackendTrain {

    private Map<TensorflowModel, Integer> global_step = new HashMap<>();
    private SessionOptions sessionOptions;
    private Session session;


    @Override
    public void delete(BackendModel m) {
        TensorflowModel model = (TensorflowModel) m;
        Status status = model.getSession().Close();
        checkStatus(status);
    }

    public BackendModel buildModel(String name, ModelParams params) throws Exception {
       Client c = new Client("localhost", 50051);
       MetaGraphDef meta = c.BuildNetwork(name, params);
       TensorflowModel model = ModelFactory.LoadFromMetaGraphDef(meta);
        int width = (int) params.get("width");
        int height = (int) params.get("height");
        int channels = (int) params.get("channels");
        int frameSize = width * height * channels;
        int classes = (Integer) params.get("classes");
        return setupModel(frameSize, classes, model);
    }

    @Override
    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts,
                  BackendParams backend_params, int num_classes, String name)
    {
        TensorflowModel model;

        if (new File(name).exists()) {

            model = ModelFactory.LoadModelFromFile(name);

        } else {

            String modelName = name + '_' + dataset.getWidth() + "x" +
                    dataset.getHeight() + "x" + dataset.getChannels() + "_" + dataset.getNumClasses();
            String resourceModelName = ModelFactory.convertToCanonicalName(modelName);
            try {
                resourceModelName = ModelFactory.findResource(resourceModelName);
            } catch (IOException e) {
                e.printStackTrace();
            }
            model = ModelFactory.LoadModelFromFile(resourceModelName);
        }

        int frameSize = dataset.getWidth() * dataset.getHeight() * dataset.getChannels();
        return setupModel(frameSize, num_classes, model);
    }

    private BackendModel setupModel(int frameSize, int num_classes, TensorflowModel model) {
        sessionOptions = new SessionOptions();
        session = new Session(sessionOptions);

        Status status = session.Create(model.getGraph());
        checkStatus(status);

        model.frameSize = frameSize;
        model.classes = num_classes;
        if (!model.meta.init.isEmpty()) {
            TensorVector outputs = new TensorVector();
            status = session.Run(
                    new StringTensorPairVector(
                            new String[]{},
                            new Tensor[]{}
                    ),
                    new StringVector(),
                    new StringVector(model.meta.init),
                    outputs);

            checkStatus(status);
        }
        model.setSession(session);
        return model;
    }

    @Override
    public void saveModel(BackendModel m, String model_path) {
        TensorflowModel model = (TensorflowModel) m;
        try {
            model.saveModel(model_path);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void loadParam(BackendModel m, String param_path) {
        // here the model should have already a pre-set of
        // graph operations that will handle the loading
        TensorVector outputs = new TensorVector();
        TensorflowModel model = (TensorflowModel) m;
        Tensor model_path_t = new Tensor(DT_STRING, new TensorShape(1));
        StringArray a = model_path_t.createStringArray();
        for (int i = 0; i < a.capacity(); i++) {
            a.position(i).put(param_path);
        }

        Status status = model.getSession().Run(
                new StringTensorPairVector(
                        new String[]{model.meta.save_filename},
                        new Tensor[]{model_path_t}
                ),
                new StringVector(),
                new StringVector(model.meta.restore_op),
                outputs
        );

        checkStatus(status);
    }

    @Override
    public void saveParam(BackendModel m, String param_path) {
        TensorVector outputs = new TensorVector();
        TensorflowModel model = (TensorflowModel) m;
        Tensor model_path_t = new Tensor(DT_STRING, new TensorShape(1));
        StringArray a = model_path_t.createStringArray();
        for (int i = 0; i < a.capacity(); i++) {
            a.position(i).put(param_path);
        }

        Status status = model.getSession().Run(
                new StringTensorPairVector(
                        new String[]{model.meta.save_filename},
                        new Tensor[]{model_path_t}
                ),
                new StringVector(model.meta.save_op),
                new StringVector(),
                outputs
        );

        checkStatus(status);
    }

    @Override
    public float[] loadMeanImage(BackendModel m, String path) {
        return new float[0];
    }

    @Override
    public String toJson(BackendModel m) {
        return null;
    }

    @Override
    public void setParameter(BackendModel m, String name, float value) {
        TensorflowModel model = (TensorflowModel) m;
        model.setParameter(name, value);
    }

    @Override
    public float[] train(BackendModel m, float[] data, float[] labels) {
        TensorflowModel model = (TensorflowModel) m;
        TensorVector outputs = new TensorVector();
        final long batchSize = data.length / model.frameSize;

        long [] labelShape = new long[]{ batchSize, model.classes};
        float [] labelData = labels;

        if (model.classes > 1) { //one-hot encoder
            labelData = new float[Math.toIntExact(batchSize * model.classes)];
            for (int i = 0; i < batchSize; i++) {
                int idx = (int)labels[i];
                labelData[i * model.classes + idx] = (float) 1.0;
            }

            labelShape = new long[]{ batchSize, model.classes };
        }

        if (!global_step.containsKey(model)) {
            global_step.put(model, 0);
        }


        Integer step = global_step.get(model);

        StringTensorPairVector feedDict = convertData(
                new float[][]{data, labelData, new float[]{step.floatValue()}},
                new long[][]{new long[]{batchSize, model.frameSize}, labelShape, new long[]{1}},
                new String[]{
                        model.meta.inputs.get("batch_image_input"),
                        model.meta.inputs.get("categorical_labels"),
                        model.meta.parameters.get("global_step"),
                }
        );

        Status status = model.getSession().Run(
                feedDict,
                new StringVector(model.meta.metrics.get("accuracy"), model.meta.metrics.get("total_loss")),
                new StringVector(model.meta.train_op),
                outputs
        );

        checkStatus(status);
        global_step.put(model, step + 1);

        return flatten(outputs);
    }

    @Override
    public float[] predict(BackendModel m, float[] data, float[] labels) {
        TensorflowModel model = (TensorflowModel) m;
        TensorVector outputs = new TensorVector();
        final long batchSize = data.length / model.frameSize;

        float[] labelData = labels;
        long[] labelShape = new long[]{batchSize, model.classes};

        if (model.classes > 1) { //one-hot encoder
            labelData = new float[Math.toIntExact(batchSize * model.classes)];
            for (int i = 0; i < batchSize; i++) {
                int idx = (int) labels[i];
                labelData[i * model.classes + idx] = (float) 1.0;
            }

            labelShape = new long[]{batchSize, model.classes};
        }

        StringTensorPairVector feedDict = convertData(
                new float[][]{data, labelData},
                new long[][]{new long[]{batchSize, model.frameSize}, labelShape},
                new String[]{ model.meta.inputs.get("batch_image_input"), model.meta.inputs.get("categorical_labels")}
        );

        Status status = model.getSession().Run(
                feedDict,
                new StringVector(model.meta.metrics.get("accuracy")),
                new StringVector(),
                outputs
        );

        checkStatus(status);

        return flatten(outputs);
    }

    private void checkStatus(Status status) {
        if (!status.ok()){
            throw new InternalError(status.error_message().getString());
        }
    }

    @Override
    public float[] predict(BackendModel m, float[] data) {
        TensorflowModel model = (TensorflowModel) m;
        TensorVector outputs = new TensorVector();
        final long batchSize = data.length / model.frameSize;

        StringTensorPairVector feedDict = convertData(
                new float[][]{data, },
                new long[][]{ new long[]{batchSize, model.frameSize}, },
                new String[]{model.meta.inputs.get("batch_image_input"), }
        );

        Status status = model.getSession().Run(
                feedDict,
                new StringVector(model.meta.predict_op), //model.meta.predict_op, model.meta.accuracy, model.meta.total_loss),
                new StringVector(),
                outputs
        );

        checkStatus(status);
       return flatten(outputs);
    }

    private float[] convertFloat(Tensor in) {
        FloatBuffer buffer = in.createBuffer();
        float[] result = new float[buffer.limit()];
        buffer.get(result);
        return result;
    }

    private long[] convertLong(Tensor in) {
        LongBuffer buffer = in.createBuffer();
        long[] result = new long[buffer.limit()];
        buffer.get(result);
        return result;
    }

    /**
        Converts a list of float arrays with shape *shapes* into a list of tensors with tensors name
     */
    private StringTensorPairVector convertData(float[][] inputs, long[][] shapes, String[] tensors){
        assert inputs.length == shapes.length:  shapes.length;
        assert inputs.length == tensors.length: inputs.length;

        Tensor[] t = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            System.out.println(shapes[i]);
            Tensor data_t = new Tensor(DT_FLOAT, new TensorShape(shapes[i]));
            FloatBuffer data_flat =  data_t.createBuffer();
            data_flat.put(inputs[i]);
            t[i] = data_t;
        }

        return new StringTensorPairVector(tensors, t);
    }

    private float[] toFloatArray(long[] arr) {
        if (arr == null) return null;
        int n = arr.length;
        float[] ret = new float[n];
        for (int i = 0; i < n; i++) {
            ret[i] = (float)arr[i];
        }
        return ret;
    }

    private float[] flatten(TensorVector arr) {
        if (arr == null) return null;
        int n = (int)arr.size();
        float[][] ret = new float[n][];
        for (int i = 0; i < n; i++) {
            switch(arr.get(i).dtype()){
                case DT_FLOAT:
                    ret[i] = convertFloat(arr.get(i));
                    if (ret[i] == null){
                        ret[i] = new float[]{};
                    }
                    continue;
                case DT_INT64:
                case DT_INT32:
                    ret[i] = toFloatArray(convertLong(arr.get(i)));
                    if (ret[i] == null){
                        ret[i] = new float[]{};
                    }
                    continue;
                default:
                    //throw new DeepwaterBackendException("unsupported");
                    System.err.println("dtype not supported:" + arr.get(i).dtype());
                    if (ret[i] == null){
                        ret[i] = new float[]{};
                    }
            }
        }
        return flattenFloat(ret);
    }

    private float[] flattenFloat(float[]... args) {
        if (args == null) return null;
        return Floats.concat(args);
    }
}
