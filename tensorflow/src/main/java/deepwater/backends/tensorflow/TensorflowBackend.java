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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;


public class TensorflowBackend implements BackendTrain {

    private static final String TMP_FOLDER = "/tmp";

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

        String paramUUID = param_path.substring(param_path.lastIndexOf("/"));

        File[] extractedFiles = ZipUtils.extractFiles(param_path, TMP_FOLDER);

        String pattern = new File(param_path).getName();
        for (File f: extractedFiles) {
            String[] partsA = pattern.split("\\.");
            String[] partsB = f.getName().split("\\.");
            for (int i = 0; i < Math.min(partsA.length, partsB.length); i++) {
                partsB[i] = partsA[i];
            }
            String newName = String.join(".", partsB);
            File newFile = new File(f.getParentFile(), newName);
            f.renameTo(newFile);
            System.out.println("renaming "+f+" to "+newFile);
        }

        String unpackedParamFiles = TMP_FOLDER + paramUUID;

        runner.feed(normalize(model.meta.save_filename), Tensor.create(unpackedParamFiles.getBytes()));
        runner.addTarget(normalize(model.meta.restore_op));
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
        runner.addTarget(normalize(model.meta.save_op));
        runner.fetch(normalize(model.meta.save_op));
        runner.run();
        File tempFile = new File(param_path);
        File tmpDir = tempFile.getParentFile();
        // Tensorflow generated files
        File[] files = listFilesWithPrefix(tmpDir, tempFile.getName());
        // Save to zip
        ZipUtils.zipFiles(new File(param_path), files);
        // Cleanup
        for (File f: files) {
           f.delete();
        }
        assert new File(param_path).exists(): "saveParam did not save. could not find file:" + param_path;
    }

    @Override
    public int paramHash(byte[] param) {
        BufferedOutputStream writer = null;
        try {
            Path tmpOutput = Files.createTempDirectory(new File(TMP_FOLDER).toPath(), UUID.randomUUID().toString());
            String zipFileName = tmpOutput.toString() + "params.zip";

            writer = new BufferedOutputStream(new FileOutputStream(zipFileName));
            writer.write(param);

            int paramHash = 1;

            ZipFile zipFile = new ZipFile(zipFileName);
            Enumeration<? extends ZipEntry> entries = zipFile.entries();
            while(entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                InputStream is = zipFile.getInputStream(entry);
                byte[] content = new byte[(int)entry.getSize()];
                is.read(content);
                is.close();
                paramHash = 31 * paramHash + Arrays.hashCode(content);
            }

            tmpOutput.toFile().delete();

            return paramHash;
        } catch (IOException e) {
            // ignore
        } finally {
            if(null != writer) {
                try {
                    writer.close();
                } catch (IOException e) {
                    // ignore
                }
            }
        }

        return -1;
    }

    private static File[] listFilesWithPrefix(File dir, String prefix){
        assert dir.exists(): "directory:"+dir+" does not exists";
        File[] foundFiles = dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.startsWith(prefix) && !name.equals(prefix);
            }
        });
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
        final long batchSize = model.miniBatchSize;

        assert data.length == model.frameSize * batchSize : "input data length is not equal to expected value";

        long[] labelShape = new long[]{batchSize, model.classes};
        float[][] labelData;

        if (model.classes > 1) { //one-hot encoder
            labelData = new float[Math.toIntExact(batchSize)][model.classes];
            for (int i = 0; i < batchSize; i++) {
                int idx = (int) labels[i];
                labelData[i][idx] = 1.0f;
            }
        } else {
            labelData = new float[Math.toIntExact(batchSize)][1];
            for (int i = 0; i < batchSize; i++) {
                assert labels.length == batchSize;
                labelData[i][0] = labels[i];
            }
        }

        float[][] dataMatrix = new float[Math.toIntExact(batchSize)][model.frameSize];
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
        //        tensors.get(0).floatValue(),
        //        tensors.get(1).floatValue()
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
