package deepwater.backends.tensorflow;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TensorflowModel;
import deepwater.datasets.ImageDataSet;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.File;
import java.io.IOException;
import java.util.List;


public class TensorflowBackend implements BackendTrain {

    static { //only load libraries once
        try {
            LibraryLoader.loadNativeLib("tensorflow_jni");
        } catch (IOException e) {
            e.printStackTrace();
            throw new IllegalArgumentException("Couldn't load tensorflow libraries");
        }
    }

    //private Map<TensorflowModel, Integer> global_step = new HashMap<>();
    //private SessionOptions sessionOptions;
    private Session session;

    @Override
    public void delete(BackendModel m) {
        TensorflowModel model = (TensorflowModel) m;
        model.setSession(null);
    }

    @Override
    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts,
                  BackendParams bparms, int num_classes, String name)
    {
        TensorflowModel model;

        int width = Math.max(dataset.getWidth(), 1);
        int height = Math.max(dataset.getHeight(), 1);
        int channels = Math.max(dataset.getChannels(), 1);
        num_classes = Math.max(num_classes, 1);

        if (new File(name).exists()) {

            model = ModelFactory.LoadModelFromFile(name);

        } else {

            String modelName = name.toLowerCase() + '_' + width + "x" +
                    height + "x" + channels + "_" + num_classes;
            String resourceModelName = ModelFactory.convertToCanonicalName(modelName);
            try {
                resourceModelName = ModelFactory.findResource(resourceModelName);
            } catch (IOException e) {
                e.printStackTrace();
            }
            model = ModelFactory.LoadModelFromFile(resourceModelName);
        }

//        sessionOptions = new SessionOptions();
//        sessionOptions.config().gpu_options().set_allow_growth(true);
//        sessionOptions.config().set_allow_soft_placement(true);
        //sessionOptions.config().set_log_device_placement(true);
        session = new Session(model.getGraph());
//        this.session = new Session(sessionOptions);


        model.frameSize = width * height * channels;
        model.classes = num_classes;
        model.miniBatchSize = (int) bparms.get("mini_batch_size");

        if (name.toLowerCase().equals("mlp")) {
            model.activations = (String[]) bparms.get("activations", new String[]{"relu"});
            model.inputDropoutRatio = (Double) bparms.get("input_dropout_ratio", 0.2);
            model.hiddenDropoutRatios = (double[]) bparms.get("hidden_dropout_ratios", new double[]{0.5});
        }

        if (!model.meta.init.isEmpty()) {
            Session.Runner runner = session.runner();
            runner.addTarget(model.meta.init).run();
        } else {
            System.out.println("WARNING WARNING: no init operation found");
            assert false;
        }
        model.setSession(this.session);
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
        TensorflowModel model = (TensorflowModel) m;

        Session.Runner runner = model.getSession().runner();

        assert new File(param_path).exists(): "cannot load from an invalid file:" + param_path;

        runner.feed(normalize(model.meta.save_filename), Tensor.create(param_path.getBytes()));
        runner.addTarget(model.meta.restore_op);
        runner.run();
    }

    private String normalize(String name){
       return name.split(":")[0];
    }

    @Override
    public void saveParam(BackendModel m, String param_path) {
        TensorflowModel model = (TensorflowModel) m;
        Session.Runner runner = model.getSession().runner();
        runner.feed(normalize(model.meta.save_filename), Tensor.create(param_path.getBytes()));
        //runner.addTarget(normalize(model.meta.save_op));
        runner.fetch(normalize(model.meta.save_op));
        runner.run();
        assert new File(param_path).exists(): "saveParam di not save. could not find file:" + param_path;
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
        if (name.equals("learning_rate")) {
            model.setParameter("learning_rate", value);
        }
        else if (name.equals("momentum")) {
            model.setParameter("momentum", value);
        }
    }

    @Override
    public float[] train(BackendModel m, float[] data, float[] labels) {
        TensorflowModel model = (TensorflowModel) m;
        final long batchSize = model.miniBatchSize;

        assert data.length == model.frameSize * batchSize: "input data length is not equal to expected value";

        long [] labelShape = new long[]{ batchSize, model.classes};
        float [][] labelData;

        if (model.classes > 1) { //one-hot encoder
            labelData = new float[Math.toIntExact(batchSize)][model.classes];
            for (int i = 0; i < batchSize; i++) {
                int idx = (int)labels[i];
                labelData[i][idx] = 1.0f;
            }
        } else {
            labelData = new float[Math.toIntExact(batchSize)][1];
            for (int i = 0; i < batchSize; i++) {
                assert labels.length == batchSize;
                labelData[i][0] = labels[i];
            }
        }

        float [][]dataMatrix = new float[Math.toIntExact(batchSize)][model.frameSize];
        int start = 0;
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(data, start, dataMatrix[i], 0, model.frameSize);
            start += model.frameSize;
        }

        Session.Runner runner = model.getSession().runner();

        runner.feed(normalize(model.meta.inputs.get("batch_image_input")), Tensor.create(dataMatrix));
        runner.feed(normalize(model.meta.inputs.get("categorical_labels")), Tensor.create(labelData));

        runner.feed(normalize(model.meta.parameters.get("learning_rate")), Tensor.create(model.getParameter("learning_rate", 0.1f)));
        runner.feed(normalize(model.meta.parameters.get("momentum")), Tensor.create(model.getParameter("momentum", 0.8f)));

//        StringTensorPairVector feedDict = convertData(
//                new float[][]{
//                        data, /* batch_image_input */
//                        labelData, /* categorical_labels */
//                        new float[]{model.getParameter("learning_rate", 0.1f)},
//                        new float[]{model.getParameter("momentum", 0.8f)},
//                },
//                new long[][]{
//                        new long[]{batchSize, model.frameSize}, /* batch_image_input */
//                        labelShape,    /* categorical_labels */
//                        new long[]{1}, /* learning_rate */
//                        new long[]{1}, /* momentum */
//                },
//                new String[]{
//                        model.meta.inputs.get("batch_image_input"),
//                        model.meta.inputs.get("categorical_labels"),
//                        model.meta.parameters.get("learning_rate"),
//                        model.meta.parameters.get("momentum")
//                }
//        );

        //runner.fetch(normalize(model.meta.metrics.get("accuracy")));
        //runner.fetch(normalize(model.meta.metrics.get("total_loss")));
        runner.addTarget(normalize(model.meta.train_op));
        List<Tensor> tensors = runner.run();

        return new float[]{
//                tensors.get(0).floatValue(),
//                tensors.get(1).floatValue()
        };
    }

    @Override
    public float[] predict(BackendModel m, float[] data, float[] labels) {
        TensorflowModel model = (TensorflowModel) m;
        final long batchSize = model.miniBatchSize;

        assert data.length == model.frameSize * batchSize: (
                " input data length " + data.length +
                " is not equal to expected value:" +
                model.frameSize * batchSize
        );
        assert labels.length == batchSize:
                ("input labels " + labels.length +
                   " is not equal to expected value: " + batchSize
        );

        float [][]dataMatrix = new float[Math.toIntExact(batchSize)][model.frameSize];
        int start = 0;
        for (int i = 0; i < batchSize; i++) {
           System.arraycopy(data, start, dataMatrix[i], 0, model.frameSize);
           start += model.frameSize;
        }

        Session.Runner runner = model.getSession().runner();

        runner.feed(normalize(model.meta.inputs.get("batch_image_input")), Tensor.create(dataMatrix));

        runner.fetch(normalize(model.meta.predict_op));
        List<Tensor> results = runner.run();
        float[][] predictions = new float[Math.toIntExact(batchSize)][model.classes];
        results.get(0).copyTo(predictions);

        float[] flatten = new float[Math.toIntExact(batchSize) * model.classes];

        start = 0;
        int length = model.classes;
        for (int i = 0; i < predictions.length; i++) {
            System.arraycopy(predictions[i], 0, flatten, start, length);
            start += model.classes;
        }
        return flatten;
    }

    @Override
    public float[] predict(BackendModel m, float[] data) {
        TensorflowModel model = (TensorflowModel) m;
        final long batchSize = model.miniBatchSize;

        assert data.length == model.frameSize * batchSize: "input data length is not equal to expected value";

        float [][]dataMatrix = new float[Math.toIntExact(batchSize)][model.frameSize];
        int start = 0;
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(data, start, dataMatrix[i], 0, model.frameSize);
            start += model.frameSize;
        }

        Session session = model.getSession();
        Session.Runner runner = session.runner();
        runner.feed(normalize(model.meta.inputs.get("batch_image_input")), Tensor.create(dataMatrix));
        runner.fetch(normalize(model.meta.predict_op));

        List<Tensor> results = runner.run();

        // extract predictions
        float[][] predictions = new float[Math.toIntExact(batchSize)][model.classes];
        results.get(0).copyTo(predictions);
        float[] flatten = new float[Math.toIntExact(batchSize) * model.classes];

        start = 0;
        int length = model.classes;
        for (int i = 0; i < predictions.length; i++) {
            System.arraycopy(predictions[i], 0, flatten, start, length);
            start += model.classes;
        }
        return flatten;
    }

}
