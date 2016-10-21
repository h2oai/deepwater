package deepwater.backends.tensorflow.test;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.TensorflowBackend;
import deepwater.datasets.BatchIterator;
import deepwater.datasets.CIFAR10ImageDataset;
import deepwater.datasets.ImageBatch;
import deepwater.datasets.MNISTImageDataset;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

public class BackendInterfaceTest {

    public float testMXnet(BackendModel model) throws IOException {
        BackendTrain backend = new TensorflowBackend();
        MNISTImageDataset dataset = new MNISTImageDataset();
        String[] images = new String[]{
                "/home/fmilo/workspace/h2o-3/t10k-images-idx3-ubyte.gz",
                "/home/fmilo/workspace/h2o-3/t10k-labels-idx1-ubyte.gz"
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
        for (int k = 0; k < loss.length; k++) {
            System.out.print(loss[k]+",");
        }
        System.out.println();
    }

    String[] mnistTrainData = new String[] {
            "/home/fmilo/workspace/h2o-3/train-images-idx3-ubyte.gz",
            "/home/fmilo/workspace/h2o-3/train-labels-idx1-ubyte.gz",
    };

    @Test
    public void testLenet() throws IOException{
        backendCanTrainMNIST("lenet", 1024, 10);
        backendCanSaveCheckpoint("lenet", 1024);
    }

    @Ignore
    @Test
    public void testVGG16() throws IOException{
        backendCanSaveCheckpoint("vgg16", 16);
        backendCanTrainMNIST("vgg16", 16, 100);
        backendCanTrainCifar10("vgg16", 16, 10);
    }

    public void backendCanTrainMNIST(String modelName, int batchSize, int epochs) throws IOException {
        BackendTrain backend = new TensorflowBackend();

        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params,
                                            dataset.getNumClasses(), modelName);

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

    public void backendCanTrainCifar10(String modelName, int batchSize, int epochs) throws IOException {
        BackendTrain backend = new TensorflowBackend();

        CIFAR10ImageDataset dataset = new CIFAR10ImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);

        String[] train_images = new String[]{
                "/datasets/cifar-10-batches-bin/data_batch_1.bin",
                "/datasets/cifar-10-batches-bin/data_batch_2.bin",
                "/datasets/cifar-10-batches-bin/data_batch_3.bin",
                "/datasets/cifar-10-batches-bin/data_batch_4.bin",
                "/datasets/cifar-10-batches-bin/data_batch_5.bin",
        };

        String[] test_images = new String[]{
                "/datasets/cifar-10-batches-bin/test_batch.bin",
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


    public void backendCanSaveCheckpoint(String modelName, int batchSize) throws IOException {

        BackendTrain backend = new TensorflowBackend();

        String[] train_images = new String[]{
                "/home/fmilo/workspace/h2o-3/train-images-idx3-ubyte.gz",
                "/home/fmilo/workspace/h2o-3/train-labels-idx1-ubyte.gz",
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
        File temp = File.createTempFile("test", ".tmp");
        backend.saveModel(model, temp.getAbsolutePath() );

        BackendModel model2 = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);
        backend.loadParam(model2, temp.getAbsolutePath() );
        System.out.println("Temp file : " + temp.getAbsolutePath());

        float loaded = testMXnet(model2);

        assert initial < trained: initial;
        assert trained <= loaded: loaded;

        backend.delete(model);
        backend.delete(model2);

    }
}
