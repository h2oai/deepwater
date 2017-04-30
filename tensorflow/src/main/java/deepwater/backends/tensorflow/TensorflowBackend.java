package deepwater.backends.tensorflow;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TensorflowModel;
import deepwater.backends.tensorflow.python.TFPythonWrapper;
import deepwater.datasets.ImageDataSet;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;

import java.io.*;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.*;


public class TensorflowBackend implements BackendTrain {

    static { //only load libraries once
        try {
            LibraryLoader.loadNativeLib("tensorflow_jni");
        } catch (IOException e) {
            e.printStackTrace();
            throw new IllegalArgumentException("Couldn't load tensorflow libraries");
        }
    }

    private Session session;

    @Override
    public void delete(BackendModel m) {
        TensorflowModel model = (TensorflowModel) m;
        model.getSession().close();
        model.setSession(null);
        model.getGraph().close();
        model.setGraph(null);
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
            } catch (Exception e) {
                resourceModelName = TFPythonWrapper.generateMetaFile(name, width, height, channels, num_classes, (int[]) bparms.get("hidden"));
            }
            model = ModelFactory.LoadModelFromFile(resourceModelName);
        }

        ConfigProto.Builder configBuilder = org.tensorflow.framework.ConfigProto.newBuilder()
                .setAllowSoftPlacement(true) // allow less GPUs than configured
                .setGpuOptions(GPUOptions.newBuilder().setAllowGrowth(true)); // don't grab all GPU RAM at once
        if(!opts.useGPU())
            configBuilder.putAllDeviceCount(Collections.singletonMap("GPU", 0));
        byte[] sessionConfig = configBuilder.build().toByteArray();

        session = new Session(model.getGraph(), sessionConfig);

        model.frameSize = width * height * channels;
        model.classes = num_classes;
        model.miniBatchSize = (int) bparms.get("mini_batch_size");

        if (name.toLowerCase().equals("mlp")) {
            model.activations = activations(bparms);
            model.inputDropoutRatio = ((Double) bparms.get("input_dropout_ratio", 0.0d)).floatValue();
            double[] hidden_dropout_ratios = hiddenDropoutratios(bparms);
            model.hiddenDropoutRatios = Floats.toArray(Doubles.asList(hidden_dropout_ratios));
        }

        if (!model.meta.init.isEmpty()) {
            Session.Runner runner = session.runner();
            Tensor isTrainingTensor = Tensor.create(false);
            feedIfPresent(runner, model.meta.parameters.get("global_is_training"), isTrainingTensor);
            runner.addTarget(model.meta.init).run();
            isTrainingTensor.close();
        } else {
            System.out.println("ERROR: no init operation found");
            return null;
        }
        model.setSession(this.session);
        return model;
    }

    // returns doubles not floats b/c we get doubles from H2O
    private double[] hiddenDropoutratios(BackendParams bparms) {
        double[] hiddenDropoutRatios = (double[]) bparms.get("hidden_dropout_ratios");
        if(null != hiddenDropoutRatios) {
            return hiddenDropoutRatios;
        }
        int layerNr = ((int[]) bparms.get("hidden")).length;
        hiddenDropoutRatios = new double[layerNr];
        for(int i = 0; i < layerNr; i++) {
            hiddenDropoutRatios[i] = 0d;
        }
        return hiddenDropoutRatios;
    }

    private String[] activations(BackendParams bparms) {
        String[] params = (String[]) bparms.get("activations");
        if(null != params) {
            return params;
        }
        int layerNr = ((int[]) bparms.get("hidden")).length;
        params = new String[layerNr];
        for(int i = 0; i < layerNr; i++) {
            params[i] = "relu";
        }

        return params;
    }

    private void feedIfPresent(Session.Runner runner, String value, Tensor tensor) {
        if(null != value) {
            runner.feed(normalize(value), tensor);
        }
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

        try(Tensor filenameTensor = Tensor.create(param_path.getBytes());
            Tensor isTrainingTensor = Tensor.create(false)){

            feedIfPresent(runner, model.meta.save_filename, filenameTensor);
            feedIfPresent(runner, model.meta.parameters.get("global_is_training"), isTrainingTensor);
            runner.addTarget(normalize(model.meta.restore_op));
            runner.run();
        }
    }

    public void writeBytes(File file, byte[] payload) throws IOException {
        ByteBuffer bb = ByteBuffer.wrap(payload);

        while(bb.hasRemaining()) {
            // Read file extension
            int extensionSize = bb.getInt();
            byte[] extension = new byte[extensionSize];
            bb.get(extension);
            String fileExtension = new String(extension);

            // Read file content
            int contentSize = bb.getInt();
            File paramFile = new File(file.getAbsolutePath() + "." + fileExtension);
            FileOutputStream fos = new FileOutputStream(paramFile);
            try {
                fos.write(bb.array(), bb.position(), contentSize);
                bb.position(bb.position() + contentSize);
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                fos.close();
            }
        }
    }

    private String normalize(String name){
        return name.split(":")[0];
    }

    @Override
    public void saveParam(BackendModel m, String param_path) {
        // Run the model to save the parameters to TF files
        TensorflowModel model = (TensorflowModel) m;
        Session.Runner runner = model.getSession().runner();

        try(Tensor filenameTensor = Tensor.create(param_path.getBytes());
            Tensor isTrainingTensor = Tensor.create(false)) {

            feedIfPresent(runner, model.meta.save_filename, filenameTensor);
            feedIfPresent(runner, model.meta.parameters.get("global_is_training"), isTrainingTensor);
            runner.addTarget(normalize(model.meta.save_op));
            runner.fetch(normalize(model.meta.save_op));
            runner.run();
        }

        File file = new File(param_path);
        File[] files = listFilesWithPrefix(file.getParentFile(), file.getName());

        assert files.length >= 2: "saveParam did not save. could not find files starting with prefix:" + param_path;
    }

    @Override
    public void deleteSavedModel(String model_path) {
        deleteAllWithPrefix(model_path);
    }

    private void deleteAllWithPrefix(String prefix) {
        File filePattern = new File(prefix);
        for(File file : listFilesWithPrefix(filePattern.getParentFile(),filePattern.getName())) {
            file.delete();
        }
    }

    @Override
    public void deleteSavedParam(String param_path) {
        deleteAllWithPrefix(param_path);
    }

    @Override
    public String listAllLayers(BackendModel m) {
        TensorflowModel model = (TensorflowModel) m;
        // This can be changed to model.getGraph().getOperations() when TF Java API implements it
        return model.meta.outputs.get("layers");
    }

    @Override
    public float[] extractLayer(BackendModel m, String name, float[] data) {
        TensorflowModel model = (TensorflowModel) m;
        Session.Runner runner = model.getSession().runner();

        assert null != model.getGraph().operation(name): "no layer with name: " + name;

        FloatBuffer dataMatrix = model.createDataMatrix(data);
        long[] dataShape = {model.miniBatchSize, model.frameSize};

        try(Tensor dataTensor = Tensor.create(dataShape, dataMatrix)) {
            runner.feed(normalize(model.meta.inputs.get("batch_image_input")), dataTensor);
            runner.fetch(name);

            Tensor run = runner.run().get(0);

            long[] shape = run.shape();
            int[] dims = new int[shape.length];
            int flatten = 1;
            for (int i = 0; i < shape.length; i++) {
                long dim = shape[i];
                dims[i] = (int) dim;
                flatten *= dim;
            }

            Object[] original = (Object[]) Array.newInstance(float.class, dims);

            run.copyTo(original);

            float[] flattened = new float[flatten];
            flatten(original, flattened, 0);
            return flattened;
        }
    }

    private int flatten(Object original, float[] flattened, int dstPos) {
        if(original instanceof float[][]) {
            for(float[] sub : (float[][]) original) {
                System.arraycopy(sub, 0, flattened, dstPos, sub.length);
                dstPos += sub.length;
            }
            return dstPos;
        } else {
            for(Object sub : (Object[]) original) {
                dstPos = flatten(sub, flattened, dstPos);
            }
        }
        return 0;
    }

    public byte[] readBytes(File filesPattern) throws IOException {
        // Read all TF files into a byte[]
        File tmpDir = filesPattern.getParentFile();
        // TF generated files
        File[] files = listFilesWithPrefix(tmpDir, filesPattern.getName());
        // Calculate required space
        int totalParamSize = 0; // needs reimplementation if all the params are bigger than Integer.MAX_VALUE
        for(File file : files) {
            totalParamSize += 4; // file extension size space
            String name = file.getName();
            totalParamSize += name.substring(name.lastIndexOf(".") + 1).getBytes().length; // file extension space

            totalParamSize += 4; // file content size space
            totalParamSize += file.length(); // file content space
        }

        // Write params to byte array. Will fail if total param size > Integet.MAX_VALUE
        byte[] params = new byte[totalParamSize];
        ByteBuffer bb = ByteBuffer.wrap(params);
        for(File file : files) {
            byte[] extension = file.getName().substring(file.getName().lastIndexOf(".") + 1).getBytes();
            int extensionLength = extension.length;

            bb.putInt(extensionLength);
            bb.put(extension);

            try {
                bb.putInt((int)file.length());
                FileInputStream f = new FileInputStream(file);
                FileChannel ch = f.getChannel();
                ch.read(bb);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return params;
    }

    private static File[] listFilesWithPrefix(File dir, String prefix){
        assert dir.exists(): "directory: "+dir+" does not exist.";
        File[] foundFiles = dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.startsWith(prefix); //&& !name.equals(prefix);
            }
        });
        Arrays.sort(foundFiles);
        return foundFiles;
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
        final int batchSize = model.miniBatchSize;
        assert data.length == model.frameSize * batchSize : "input data length is not equal to expected value";

        FloatBuffer labelData = FloatBuffer.allocate(batchSize * model.classes);
        long[] labelShape = new long[] {batchSize, model.classes};
        if (model.classes > 1) { //one-hot encoder
            for (int i = 0; i < batchSize; i++) {
                int idx = (int) labels[i];
                for(int j = 0; j < model.classes; j++) {
                    if(j == idx) {
                        labelData.put(1.0f);
                    } else {
                        labelData.put(0.0f);
                    }
                }
            }
        } else if (model.classes == 1){ //regression
            for (int i = 0; i < batchSize; i++) {
                assert labels.length == batchSize;
                labelData.put(labels[i]);
            }
        } else throw new IllegalArgumentException();
        labelData.flip();

        FloatBuffer dataMatrix = model.createDataMatrix(data);
        long[] dataShape = {batchSize, model.frameSize};

        Session.Runner runner = model.getSession().runner();

        try(Tensor dataTensor = Tensor.create(dataShape, dataMatrix);
            Tensor labelTensor = Tensor.create(labelShape, labelData);
            Tensor batchSizeTensor = Tensor.create((float) model.miniBatchSize);
            Tensor learningRateTensor = Tensor.create(model.getParameter("learning_rate", 0.001f));
            Tensor momentumTensor = Tensor.create(model.getParameter("momentum", 0.9f));
            Tensor isTrainingTensor = Tensor.create(true)) {

            runner.feed(
                    normalize(model.meta.inputs.get("batch_image_input")),
                    dataTensor);
            runner.feed(normalize(model.meta.inputs.get("categorical_labels")), labelTensor);

            List<Tensor> mlpTensors = Collections.emptyList();
            if (null != model.activations) {
                mlpTensors = feedMLPData(model, runner);
            }

            feedIfPresent(runner, model.meta.parameters.get("batch_size"), batchSizeTensor);

            feedIfPresent(runner, model.meta.parameters.get("learning_rate"), learningRateTensor);
            feedIfPresent(runner, model.meta.parameters.get("momentum"), momentumTensor);

            feedIfPresent(runner, model.meta.parameters.get("global_is_training"), isTrainingTensor);

            runner.addTarget(normalize(model.meta.train_op));

            runner.run();//nothing to fetch

            for(Tensor t : mlpTensors) {
                t.close();
            }

            return null;
        }
    }

    private List<Tensor> feedMLPData(TensorflowModel model, Session.Runner runner) {
        // String tensors not supported in this version of TF Java API
        int[] act = new int[model.activations.length];
        for(int i = 0; i < model.activations.length; i++) {
            act[i] = TensorflowModel.activationToNumeric.getOrDefault(model.activations[i], 0);
        }

        Tensor activationTensor = Tensor.create(act);
        runner.feed(normalize(model.meta.parameters.get("activations")), activationTensor);
        Tensor dropoutRatioTensor = Tensor.create(model.inputDropoutRatio);
        runner.feed(normalize(model.meta.parameters.get("input_dropout")), dropoutRatioTensor);
        Tensor hiddenDropoutRatioTensor = Tensor.create(model.hiddenDropoutRatios);
        runner.feed(normalize(model.meta.parameters.get("hidden_dropout")), hiddenDropoutRatioTensor);

        return new ArrayList<Tensor>() {{this.add(activationTensor); this.add(dropoutRatioTensor); this.add(hiddenDropoutRatioTensor);}};
    }

    @Override
    public float[] predict(BackendModel m, float[] data) {
        TensorflowModel model = (TensorflowModel) m;
        Session session = model.getSession();
        Session.Runner runner = session.runner();
        FloatBuffer dataMatrix = model.createDataMatrix(data);
        long[] dataShape = {model.miniBatchSize, model.frameSize};

        List<Tensor> mlpTensors = Collections.emptyList();
        if(null != model.activations) {
            mlpTensors = feedMLPData(model, runner);
        }

        try(Tensor miniBatchSizeTensor = Tensor.create((float) model.miniBatchSize);
            Tensor isTrainingTensor = Tensor.create(false);
            Tensor dataTensor = Tensor.create(dataShape, dataMatrix)) {

            feedIfPresent(runner, model.meta.parameters.get("batch_size"), miniBatchSizeTensor);

            feedIfPresent(runner, model.meta.parameters.get("global_is_training"), isTrainingTensor);
            feedIfPresent(runner, model.meta.inputs.get("batch_image_input"), dataTensor);
            runner.fetch(normalize(model.meta.predict_op)); //the tensor we want to extract
            List<Tensor> results = runner.run();

            for(Tensor t : mlpTensors) {
                t.close();
            }

            Tensor output = results.get(0);
            return model.getPredictions(output);
        }
    }

}
