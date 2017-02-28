package deepwater.backends.tensorflow;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TensorflowModel;
import deepwater.datasets.ImageDataSet;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.*;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
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

        session = new Session(model.getGraph());

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
            runner.feed(normalize(model.meta.parameters.get("global_is_training")), Tensor.create(false));
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

        runner.feed(normalize(model.meta.save_filename), Tensor.create(param_path.getBytes()));
        runner.feed(normalize(model.meta.parameters.get("global_is_training")), Tensor.create(false));
        runner.addTarget(normalize(model.meta.restore_op));
        runner.run();
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

        runner.feed(normalize(model.meta.save_filename), Tensor.create(param_path.getBytes()));
        runner.feed(normalize(model.meta.parameters.get("global_is_training")), Tensor.create(false));
        runner.addTarget(normalize(model.meta.save_op));
        runner.fetch(normalize(model.meta.save_op));
        runner.run();

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

        float[][] dataMatrix = model.createDataMatrix(data);
        runner.feed(normalize(model.meta.inputs.get("batch_image_input")), Tensor.create(dataMatrix));
        runner.fetch(name);

        Tensor run = runner.run().get(0);

        long[] shape = run.shape();
        int[] dims = new int[shape.length];
        int flatten = 1;
        for(int i = 0; i < shape.length; i++) {
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
        assert dir.exists(): "directory:"+dir+" does not exists";
        File[] foundFiles = dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.startsWith(prefix) && !name.equals(prefix);
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
        float[][] labelData;
        labelData = new float[batchSize][model.classes];
        if (model.classes > 1) { //one-hot encoder
            for (int i = 0; i < batchSize; i++) {
                int idx = (int) labels[i];
                labelData[i][idx] = 1.0f;
            }
        } else if (model.classes == 1){ //regression
            for (int i = 0; i < batchSize; i++) {
                assert labels.length == batchSize;
                labelData[i][0] = labels[i];
            }
        } else throw new IllegalArgumentException();

        float[][] dataMatrix = model.createDataMatrix(data);

        Session.Runner runner = model.getSession().runner();

        runner.feed(normalize(model.meta.inputs.get("batch_image_input")), Tensor.create(dataMatrix));
        runner.feed(normalize(model.meta.inputs.get("categorical_labels")), Tensor.create(labelData));

        runner.feed(normalize(model.meta.parameters.get("learning_rate")), Tensor.create(model.getParameter("learning_rate", 0.001f)));
        runner.feed(normalize(model.meta.parameters.get("momentum")), Tensor.create(model.getParameter("momentum", 0.5f)));

        runner.feed(normalize(model.meta.parameters.get("global_is_training")), Tensor.create(true));

        runner.addTarget(normalize(model.meta.train_op));

        runner.run();//nothing to fetch
        return null;
    }

    @Override
    public float[] predict(BackendModel m, float[] data) {
        TensorflowModel model = (TensorflowModel) m;
        Session session = model.getSession();
        Session.Runner runner = session.runner();
        float[][] dataMatrix = model.createDataMatrix(data);
        runner.feed(normalize(model.meta.parameters.get("global_is_training")), Tensor.create(false));
        runner.feed(normalize(model.meta.inputs.get("batch_image_input")), Tensor.create(dataMatrix));
        runner.fetch(normalize(model.meta.predict_op)); //the tensor we want to extract
        List<Tensor> results = runner.run();
        Tensor output = results.get(0);
        float[] preds = model.getPredictions(output);
        return preds;
    }

}
