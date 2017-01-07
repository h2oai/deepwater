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
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static deepwater.datasets.FileUtils.findFile;


public class BackendInterfaceTest {

    private String[] mnistTrainData = new String[]{
            findFile("bigdata/laptop/mnist/train-images-idx3-ubyte.gz"),
            findFile("bigdata/laptop/mnist/train-labels-idx1-ubyte.gz"),
    };

    private float testMXnet(BackendModel model) throws IOException {
        BackendTrain backend = new TensorflowBackend();
        MNISTImageDataset dataset = new MNISTImageDataset();
        String[] images = new String[]{
                findFile("bigdata/laptop/mnist/t10k-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/t10k-labels-idx1-ubyte.gz")
        };

        BatchIterator it = new BatchIterator(dataset, 1, images);
        ImageBatch b = new ImageBatch(dataset, 1024);

        while(it.next(b)){
            float[] loss = backend.predict(model, b.getImages(), b.getLabels());

            printLoss(loss);
            return loss[0];
        }


        return 0;
    }

    private void printLoss(float[] loss) {
        System.out.print("Test accuracy:");
        for (float los : loss) {
            System.out.print(los + ",");
        }
        System.out.println();
    }

    @Test
    public void testLenet() throws IOException{
        backendCanTrainMNIST("lenet", 32, 3);
        backendCanSaveCheckpoint("lenet", 32);
    }

    @Ignore
    @Test
    public void testVGG() throws IOException {
        backendCanSaveCheckpoint("vgg", 16);
        backendCanTrainMNIST("vgg", 32, 10);
        backendCanTrainCifar10("vgg", 32, 10);
    }

    private void backendCanTrainMNIST(String modelName, int batchSize, int epochs) throws IOException {
        BackendTrain backend = new TensorflowBackend();

        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        params.set("mini_batch_size", batchSize);

        BackendModel model = backend.buildNet(dataset, opts, params,
                                            dataset.getNumClasses(), modelName);

        backend.setParameter(model, "learning_rate", 0.1f);
        backend.setParameter(model, "momentum", 0.8f);

        BatchIterator it = new BatchIterator(dataset, epochs, mnistTrainData);
        ImageBatch b = new ImageBatch(dataset, batchSize);

        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }
            testMXnet(model);
        }
        backend.delete(model);
    }

    private void backendCanTrainCifar10(String modelName, int batchSize, int epochs) throws IOException {
        BackendTrain backend = new TensorflowBackend();

        CIFAR10ImageDataset dataset = new CIFAR10ImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);

        String[] train_images = new String[]{
                findFile("bigdata/laptop/cifar-10-batches-bin/data_batch_1.bin"),
                findFile("bigdata/laptop/cifar-10-batches-bin/data_batch_2.bin"),
                findFile("bigdata/laptop/cifar-10-batches-bin/data_batch_3.bin"),
                findFile("bigdata/laptop/cifar-10-batches-bin/data_batch_4.bin"),
                findFile("bigdata/laptop/cifar-10-batches-bin/data_batch_5.bin"),
        };

        String[] test_images = new String[]{
                findFile("bigdata/laptop/cifar-10-batches-bin/test_batch.bin"),
        };

        BatchIterator it = new BatchIterator(dataset, epochs, train_images);
        ImageBatch b = new ImageBatch(dataset, batchSize);

        BatchIterator test_it = new BatchIterator(dataset, 1, test_images);
        ImageBatch bb = new ImageBatch(dataset, batchSize);

        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }

            while (test_it.next(bb)) {
                float[] loss = backend.predict(model, bb.getImages(), bb.getLabels());
                printLoss(loss);
            }

        }

        backend.delete(model);
    }


    private void backendCanSaveCheckpoint(String modelName, int batchSize) throws IOException {

        BackendTrain backend = new TensorflowBackend();

        String[] train_images = new String[]{
                findFile("bigdata/laptop/mnist/train-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/train-labels-idx1-ubyte.gz"),
        };
        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();
        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);

        float initial = testMXnet(model);

        BatchIterator it = new BatchIterator(dataset, 1, train_images);
        ImageBatch b = new ImageBatch(dataset, batchSize);
        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }
        }
        float trained = testMXnet(model);

        //create a temp file
        File modelFile = File.createTempFile("model", ".tmp");
        backend.saveModel(model, modelFile.getAbsolutePath());

        File modelParams = File.createTempFile("params", ".tmp");
        backend.saveParam(model, modelParams.getAbsolutePath());

        BackendModel model2 = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelFile.getAbsolutePath());

        backend.loadParam(model2, modelParams.getAbsolutePath());

        float loaded = testMXnet(model2);

        assert initial < trained: initial;
        assert trained <= loaded: loaded;

        backend.delete(model);
        backend.delete(model2);

    }

    @Test
    public void backendCanLoadMetaGraph() throws Exception {
        final String meta_model = ModelFactory.findResource("my-model-20001.meta");

        MNISTImageDataset dataset = new MNISTImageDataset();

        BackendTrain backend = new TensorflowBackend();
        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params,
                dataset.getNumClasses(),
                meta_model);

        File modelFile = File.createTempFile("model", ".tmp");
        backend.saveModel(model, modelFile.getPath());

        File modelParams = File.createTempFile("params", ".tmp");
        backend.saveParam(model, modelParams.getAbsolutePath());


        opts = new RuntimeOptions();
        params = new BackendParams();

        BackendModel model2 = backend.buildNet(dataset, opts, params,
                dataset.getNumClasses(),
                modelFile.getAbsolutePath());

        backend.loadParam(model2, modelParams.getAbsolutePath());
    }
}
