package deepwater.backends.tensorflow.test;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.TensorflowBackend;
import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.datasets.BatchIterator;
import deepwater.datasets.CIFAR10ImageDataset;
import deepwater.datasets.ImageBatch;
import deepwater.datasets.MNISTImageDataset;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

import static deepwater.datasets.FileUtils.findFile;
import static java.lang.Float.NaN;


public class BackendInterfaceTest {

    private String[] mnistTrainData = new String[]{
            findFile("bigdata/laptop/mnist/train-images-idx3-ubyte.gz"),
            findFile("bigdata/laptop/mnist/train-labels-idx1-ubyte.gz"),
    };

    private double computeTestErrorMNIST(BackendModel model, int batchSize) throws IOException {
        BackendTrain backend = new TensorflowBackend();
        MNISTImageDataset dataset = new MNISTImageDataset();
        String[] images = new String[]{
                findFile("bigdata/laptop/mnist/t10k-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/t10k-labels-idx1-ubyte.gz")
        };

        BatchIterator it = new BatchIterator(dataset, 1, images);
        ImageBatch b = new ImageBatch(dataset, batchSize);

        return computeValidationError(backend, model, it, b, 10);
    }

    private double computeCIFAR10TestError(BackendModel model, int batchSize) throws IOException {
        BackendTrain backend = new TensorflowBackend();
        CIFAR10ImageDataset dataset = new CIFAR10ImageDataset();

        String[] test_images = new String[]{
                findFile("bigdata/laptop/mnist/cifar-10/test_batch.bin"),
        };
        BatchIterator test_it = new BatchIterator(dataset, 1, test_images);
        ImageBatch batchTest = new ImageBatch(dataset, batchSize);

        return computeValidationError(backend, model, test_it, batchTest, dataset.getNumClasses());
    }

    private double computeValidationError(BackendTrain backend, BackendModel model, BatchIterator it, ImageBatch b, int classNum) throws IOException {
        double error = 0.0;
        double total = 0.0;

        while(it.next(b)){
            float[] predictions = backend.predict(model, b.getImages());
            float[] labels = b.getLabels();
            for (int i = 0, j = 0; i < predictions.length; i += classNum, j++) {
                int classPrediction = argmax(predictions, i, i + classNum);
                if (classPrediction != labels[j]){
                   error++;
                }
                total++;
            }
        }

        return (error/total) * 100.0;
    }

    private double computeTestErrorCifar10(BackendModel model, int batchSize) throws IOException {
        BackendTrain backend = new TensorflowBackend();
        CIFAR10ImageDataset dataset = new CIFAR10ImageDataset();
        String[] images = new String[]{
                findFile("bigdata/laptop/mnist/cifar-10/test_batch.bin")
        };

        BatchIterator it = new BatchIterator(dataset, 1, images);
        ImageBatch b = new ImageBatch(dataset, batchSize);

        int classNum = 10;
        double error = 0.0;
        double total = 0.0;

        while(it.next(b)){
            float[] predictions = backend.predict(model, b.getImages());
            float[] labels = b.getLabels();
            for (int i = 0, j = 0; i < predictions.length; i += classNum, j++) {
                int classPrediction = argmax(predictions, i, i + 10);
                if (classPrediction != labels[j]){
                    error++;
                }
                total++;
            }
        }

        return error/total;
    }



    private int argmax(float[] values, int start, int end){
        int argmax = 0;
        float maxvalue = 0;
        for (int i = start; i < end; i++) {
           if (values[i] > maxvalue){
                argmax = i;
                maxvalue = values[i];
           }
        }
        return argmax - start;
    }

    private void printLoss(float[] loss) {
        System.out.print("Test accuracy:");
        for (float los : loss) {
            System.out.print(los + ",");
        }
        System.out.println();
    }

    @Test
    public void testMLP() throws IOException{
        backendCanTrainMNIST("mlp", 32, 1);
        backendCanSaveCheckpointMNIST("mlp", 32, 0.2f);
    }

    @Test
    public void testLenet() throws IOException{
        backendCanTrainMNIST("lenet", 32, 1);
        backendCanSaveCheckpointMNIST("lenet", 32, 0.1f);
    }

    @Test
    public void testAlexnet() throws IOException{
        backendCanTrainMNIST("alexnet", 32, 1, 0.01f);
        backendCanSaveCheckpointMNIST("alexnet", 32, 0.01f);

//        backendCanTrainCifar10("alexnet", 32, 1, 0.01f);
//        backendCanSaveCheckpointCifar10("alexnet", 32, 1, 0.05f);
    }

    @Test
    public void testVGG() throws IOException {
        backendCanTrainMNIST("vgg", 32, 2, 0.05f);
        backendCanSaveCheckpointMNIST("vgg", 64, 0.05f);

//        backendCanTrainCifar10("vgg", 1, 2, 0.01f);
//        backendCanSaveCheckpointCifar10("vgg", 32, 1, 0.05f);
    }

    @Test
    public void testInception() throws IOException {
        backendCanTrainMNIST("inception_bn", 16, 2, 0.01f);
        backendCanSaveCheckpointMNIST("inception_bn", 16, 0.01f);

//        backendCanTrainCifar10("inception_bn", 32, 1, 0.05f);
//        backendCanSaveCheckpointCifar10("inception_bn", 32, 1, 0.05f);
    }

    private void backendCanTrainMNIST(String modelName, int batchSize, int epochs) throws IOException {
        backendCanTrainMNIST(modelName, batchSize, epochs, 0.1f);
    }

    private void backendCanTrainMNIST(String modelName, int batchSize, int epochs, float learningRate) throws IOException {
        BackendTrain backend = new TensorflowBackend();

        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        params.set("mini_batch_size", batchSize);

        BackendModel model = backend.buildNet(dataset, opts, params,
                                            dataset.getNumClasses(), modelName);

        backend.setParameter(model, "learning_rate", learningRate);
        backend.setParameter(model, "momentum", 0.8f);

        double initialError = computeTestErrorMNIST(model, batchSize);

        System.out.println("initial MNIST test error:" + initialError);

        BatchIterator it = new BatchIterator(dataset, epochs, mnistTrainData);
        ImageBatch b = new ImageBatch(dataset, batchSize);

        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
                double trainError = computePredictionError(backend, model, b, dataset.getNumClasses());
                System.out.println("error:" + trainError);
            }

            learningRate *= 0.5;
            backend.setParameter(model, "learning_rate", learningRate);
        }
        double testError = computeTestErrorMNIST(model, batchSize);

        backend.delete(model);
        System.out.println("final MNIST test error:" + testError);

        assert testError < initialError: "final error is not less than initial error. model did not learn.";
    }

    private void backendCanTrainCifar10(String modelName, int batchSize, int epochs, float learningRate) throws IOException {
        BackendTrain backend = new TensorflowBackend();

        CIFAR10ImageDataset dataset = new CIFAR10ImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        params.set("mini_batch_size", batchSize);

        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);

        backend.setParameter(model, "learning_rate", learningRate);
        backend.setParameter(model, "momentum", 0.8f);

        double initialError = computeCIFAR10TestError(model, batchSize);

        System.out.println("initial CIFAR10 validation error:" + initialError);

        String[] train_images = new String[]{
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_1.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_2.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_3.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_4.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_5.bin"),
        };


        BatchIterator it = new BatchIterator(dataset, epochs, train_images);
        ImageBatch b = new ImageBatch(dataset, batchSize);

        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
                double trainError = computePredictionError(backend, model, b, dataset.getNumClasses());
                System.out.println("train error:" + trainError);
            }

            double err = computeCIFAR10TestError(model, batchSize);
            System.out.println("train error:" + err);
        }

        double error = computeCIFAR10TestError(model, batchSize);

        System.out.println("final CIFAR10 validation error:" + error);

        assert error < initialError: "final error is not less than initial error. model did not learn.";

        backend.delete(model);
    }

    private double computePredictionError(BackendTrain backend, BackendModel model, ImageBatch b, int classes) {
        double error = 0.0;
        double total = 0.0;
        float[] predictions = backend.predict(model, b.getImages());
        float[] labels = b.getLabels();
        for (int i = 0, j = 0; i < predictions.length; i += classes, j++) {
            assert predictions[i] != NaN: "Found Nan inside prediction";
            int classPrediction = argmax(predictions, i, i + classes);
            if (classPrediction != labels[j]){
                error++;
            }
            total++;
        }

        return error/total * 100.0;
    }


    private void backendCanSaveCheckpointMNIST(String modelName, int batchSize, float learningRate) throws IOException {

        BackendTrain backend = new TensorflowBackend();

        String[] train_images = new String[]{
                findFile("bigdata/laptop/mnist/train-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/train-labels-idx1-ubyte.gz"),
        };
        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();
        params.set("mini_batch_size", batchSize);
        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);

        double initial = computeTestErrorMNIST(model, batchSize);

        backend.setParameter(model, "learning_rate", learningRate);
        backend.setParameter(model, "momentum", 0.8f);

        BatchIterator it = new BatchIterator(dataset, 1, train_images);
        ImageBatch b = new ImageBatch(dataset, batchSize);
        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }
        }
        double trained = computeTestErrorMNIST(model, batchSize);

        //create a temp file
        File modelFile = File.createTempFile("model", ".tmp");
        backend.saveModel(model, modelFile.getAbsolutePath());

        File modelParams = File.createTempFile("params", ".tmp");
        backend.saveParam(model, modelParams.getAbsolutePath());

        BackendModel model2 = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelFile.getAbsolutePath());

        backend.loadParam(model2, modelParams.getAbsolutePath());

        double loaded = computeTestErrorMNIST(model2, batchSize);

        System.out.printf("error rate: initial %f  trained %f  improvement %f\n", initial, trained, initial - trained);
        assert initial > trained: ("initial error rate:" + initial + " is less or same as after being trained: " + trained);
        assert trained <= loaded: loaded;

        backend.delete(model);
        backend.delete(model2);

    }

    private void backendCanSaveCheckpointCifar10(String modelName, int batchSize, int epochs, float learningRate) throws IOException {

        BackendTrain backend = new TensorflowBackend();

        String[] train_images = new String[]{
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_1.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_2.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_3.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_4.bin"),
                findFile("bigdata/laptop/mnist/cifar-10/data_batch_5.bin"),
        };
        CIFAR10ImageDataset dataset = new CIFAR10ImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();
        params.set("mini_batch_size", batchSize);
        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);

        double initial = computeTestErrorCifar10(model, batchSize);

        backend.setParameter(model, "learning_rate", learningRate);
        backend.setParameter(model, "momentum", 0.8f);

        BatchIterator it = new BatchIterator(dataset, epochs, train_images);
        ImageBatch b = new ImageBatch(dataset, batchSize);
        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }
        }
        double trained = computeTestErrorCifar10(model, batchSize);

        //create a temp file
        File modelFile = File.createTempFile("model", ".tmp");
        backend.saveModel(model, modelFile.getAbsolutePath());

        File modelParams = File.createTempFile("params", ".tmp");
        backend.saveParam(model, modelParams.getAbsolutePath());

        BackendModel model2 = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelFile.getAbsolutePath());

        backend.loadParam(model2, modelParams.getAbsolutePath());

        double loaded = computeTestErrorCifar10(model2, batchSize);

        System.out.printf("error rate: initial %f  trained %f  improvement %f\n", initial, trained, initial - trained);
        assert initial > trained: ("initial error rate:" + initial + " is less or same as after being trained: " + trained);
        assert trained <= loaded: loaded;

        backend.delete(model);
        backend.delete(model2);

    }


    @Test
    public void backendCanLoadMetaGraph() throws Exception {
//        final String meta_model = ModelFactory.findResource("mlp_10x1x1_1.meta");
        final String meta_model = ModelFactory.findResource("vgg_224x224x3_1000.meta");

        MNISTImageDataset dataset = new MNISTImageDataset();

        BackendTrain backend = new TensorflowBackend();
        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();
        params.set("mini_batch_size", 1);
        BackendModel model = backend.buildNet(dataset, opts, params,
                dataset.getNumClasses(),
                meta_model);

        File modelFile = File.createTempFile("model", ".tmp");
        modelFile.delete();
        backend.saveModel(model, modelFile.getPath());

        File modelParams = File.createTempFile("params", ".tmp");
        modelParams.delete();
        backend.saveParam(model, modelParams.getAbsolutePath());

        opts = new RuntimeOptions();
        params = new BackendParams();
        params.set("mini_batch_size", 1);

        BackendModel model2 = backend.buildNet(dataset, opts, params,
                dataset.getNumClasses(),
                modelFile.getAbsolutePath());
        backend.loadParam(model2, modelParams.getAbsolutePath());
    }
}
