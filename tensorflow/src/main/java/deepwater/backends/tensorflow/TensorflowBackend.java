package deepwater.backends.tensorflow;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TensorflowModel;
import deepwater.datasets.ImageDataSet;

import java.nio.FloatBuffer;

import static org.bytedeco.javacpp.tensorflow.*;


public class TensorflowBackend implements BackendTrain {

    @Override
    public void delete(BackendModel m) {
        TensorflowModel model = (TensorflowModel) m;
        model.getSession().Close();
    }

    @Override
    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts,
                  BackendParams backend_params, int num_classes, String name)
    {
        TensorflowModel model = ModelFactory.LoadModel(name+'_'+dataset.getWidth()+"x"+
                dataset.getHeight()+"x"+dataset.getChannels()+"_"+dataset.getNumClasses());
        SessionOptions sessionOptions = new SessionOptions();
        Session session = new Session(sessionOptions);
        Status status = session.Create(model.getGraph());
        assert status.ok(): status.error_message().getString();
        model.frameSize = dataset.getWidth() * dataset.getHeight() * dataset.getChannels();
        TensorVector outputs = new TensorVector();
        status = session.Run(
                new StringTensorPairVector(
                        new String[]{},
                        new Tensor[]{}
                ),
                new StringVector(),
                new StringVector("init"),
                outputs);
        assert status.ok(): status.error_message().getString();
        model.setSession(session);
        return model;
    }

    @Override
    public void saveModel(BackendModel m, String model_path) {
        TensorVector outputs = new TensorVector();
        TensorflowModel model = (TensorflowModel) m;
        Tensor model_path_t = new Tensor(DT_STRING, new TensorShape(1));
        StringArray a = model_path_t.createStringArray();
        for (int i = 0; i < a.capacity(); i++) {
            a.position(i).put(model_path);
        }
        model.getSession().Run(
                new StringTensorPairVector(
                        new String[]{"save/Const:0"},
                        new Tensor[]{model_path_t}
                ),
                new StringVector(),
                new StringVector("save/control_dependency:0"),
                outputs
        );
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
                        new String[]{"save/Const:0"},
                        new Tensor[]{model_path_t}
                ),
                new StringVector(),
                new StringVector("save/restore_all"),
                outputs
        );
        if (!status.ok()){
            System.err.println("status: " + status.error_message().getString());
        }
    }

    @Override
    public void saveParam(BackendModel m, String param_path) {
        saveModel(m, param_path);
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

    }

    @Override
    public float[] train(BackendModel m, float[] data, float[] label) {
        TensorVector outputs = new TensorVector();
        TensorflowModel model = (TensorflowModel) m;

        int batchSize = data.length / model.frameSize;

        // data
        Tensor data_t = new Tensor(DT_FLOAT, new TensorShape(batchSize, data.length));
        FloatBuffer data_flat =  data_t.createBuffer();
        data_flat.put(data);

        // labels
        Tensor label_t = new Tensor(DT_FLOAT, new TensorShape(batchSize, label.length));
        FloatBuffer labels_flat =  label_t.createBuffer();
        labels_flat.put(label);

        // for (int i = 0; i < model.getGraph().node_size(); i++) {
        //     System.out.println(model.getGraph().node(i).name().getString());
        // }

        Tensor learning_rate_t = new Tensor(DT_FLOAT, new TensorShape(1));
        FloatBuffer learning_rate_flat =  learning_rate_t.createBuffer();
        learning_rate_flat.put(0.05f);

        Status status = model.getSession().Run(
                new StringTensorPairVector(
                        new String[]{"input/Placeholder:0", "train/Placeholder:0", "learning_rate" },
                        new Tensor[]{data_t, label_t, learning_rate_t}
                ),
                new StringVector("inference/predictions/Reshape_1"),
                new StringVector("OptimizeLoss/train"),
                outputs
        );

        if (!status.ok()){
            System.out.println(status.error_message().getString());
            return new float[]{};
        }

        Tensor result_t = outputs.get(0);
        FloatBuffer buffer = result_t.createBuffer();
        float[] result = new float[buffer.limit()];
        buffer.put(result);
        return result;
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
