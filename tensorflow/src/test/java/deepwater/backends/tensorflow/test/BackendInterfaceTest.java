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
        ImageBatch b = new ImageBatch(dataset, 10000);

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
    public void backendCanTrainMXNet() throws IOException {
        BackendTrain backend = new TensorflowBackend();

        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params,
                                            dataset.getNumClasses(), "simple");

        BatchIterator it = new BatchIterator(dataset, 10, mnistTrainData);
        ImageBatch b = new ImageBatch(dataset, 32);

        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }
            testMXnet(model);
        }
    }

    // canSave // can Load
    @Test
    public void backendCanTrainCifar10() throws IOException {
        BackendTrain backend = new TensorflowBackend();

        CIFAR10ImageDataset dataset = new CIFAR10ImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), "simple");

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

        BatchIterator it = new BatchIterator(dataset, 30, train_images);
        ImageBatch b = new ImageBatch(dataset, 32);

        BatchIterator test_it = new BatchIterator(dataset, 1, test_images);
        ImageBatch bb = new ImageBatch(dataset, 1024);

        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }

            while (test_it.next(bb)) {
                float[] loss = backend.predict(model, bb.getImages(), bb.getLabels());
                printLoss(loss);
            }

        }
    }


    @Test
    public void backendCanSaveCheckpoint() throws IOException {

        BackendTrain backend = new TensorflowBackend();

        String[] train_images = new String[]{
                "/home/fmilo/workspace/h2o-3/train-images-idx3-ubyte.gz",
                "/home/fmilo/workspace/h2o-3/train-labels-idx1-ubyte.gz",
        };
        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();
        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), "simple");

        float initial = testMXnet(model);

        BatchIterator it = new BatchIterator(dataset, 1, train_images);
        ImageBatch b = new ImageBatch(dataset, 32);
        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }
        }
        float trained = testMXnet(model);

        //create a temp file
        File temp = File.createTempFile("temp-file-name", ".tmp");
        backend.saveModel(model, temp.getAbsolutePath() );

        BackendModel model2 = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), "simple");
        backend.loadParam(model2, temp.getAbsolutePath() );
        System.out.println("Temp file : " + temp.getAbsolutePath());

        float loaded = testMXnet(model2);

        assert initial < trained: initial;
        assert trained <= loaded: trained;

    }
}
