package deepwater.backends.tensorflow.test;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.TensorflowBackend;
import deepwater.datasets.CIFAR10ImageDataset;
import deepwater.datasets.MNISTImageDataset;
import deepwater.datasets.Pair;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class BackendInterfaceTest {

    public void testMXnet(BackendModel model) throws IOException {
        BackendTrain backend = new TensorflowBackend();
        MNISTImageDataset dataset = new MNISTImageDataset("/home/fmilo/workspace/h2o-3/t10k-labels-idx1-ubyte.gz",
                "/home/fmilo/workspace/h2o-3/t10k-images-idx3-ubyte.gz");
        List<Pair<Integer, float[]>> mnist = dataset.loadDigitImages();

        float[] labels = new float[10 * mnist.size()];
        float[] images = new float[784 * mnist.size()];
        int i = 0;
        for (Pair<Integer, float[]> entry : mnist) {

            float[] image = entry.getSecond();
            Integer label = entry.getFirst();
            System.arraycopy(image, 0, images, i * image.length, image.length);
            labels[i * 10 + label] = (float) 1.0;
            i++;
        }

        float[] loss = backend.predict(model, images, labels);

        System.out.print("Test accuracy:");
        for (int k = 0; k < loss.length; k++) {
            System.out.print(loss[k]+",");
        }
        System.out.println();
    }

    @Test
    public void backendCanTrainMXNet() throws IOException {
        BackendTrain backend = new TensorflowBackend();

        MNISTImageDataset dataset = new MNISTImageDataset("/home/fmilo/workspace/h2o-3/train-labels-idx1-ubyte.gz",
                "/home/fmilo/workspace/h2o-3/train-images-idx3-ubyte.gz");

        List<Pair<Integer, float[]>> mnist = dataset.loadDigitImages();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params,
                                            dataset.getNumClasses(), "simple");
        final int batchSize = 1024;
        float[] labels = new float[10 * batchSize];
        float[] images = new float[784 * batchSize];

        for (int j = 0; j < 50; j++) {
                int i = 0;
                Arrays.fill(labels,0);
                Arrays.fill(images, 0);

                for (Pair<Integer, float[]> entry : mnist) {

                    float[] image = entry.getSecond();
                    Integer label = entry.getFirst();
                    System.arraycopy(image, 0, images, i * image.length, image.length);
                    labels[i * 10 + label] = (float) 1.0;
                    i++;

                    if (i == batchSize) {
                        float[] loss = backend.train(model, images, labels);
                        Arrays.fill(labels, 0);
                        Arrays.fill(images, 0);
                        i = 0;
//                        for (int k = 0; k < loss.length; k++) {
//                            System.out.print(loss[k]+",");
//                        }
//                        System.out.println();
                    }
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

        List<Pair<Integer, float[]>> images = dataset.loadImages("/datasets/cifar-10-batches-bin/data_batch_1.bin");

        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), "cifarnet");

       for( Pair<Integer, float[]> entry: images ) {
           float[] image = entry.getSecond();
           Integer label = entry.getFirst();
           float[] one_hot = new float[10];
           one_hot[label.intValue()] = (float)1.0;
           float[] loss = backend.train(model, image, one_hot );
           System.out.println(loss[0]);
       }
    }
}
