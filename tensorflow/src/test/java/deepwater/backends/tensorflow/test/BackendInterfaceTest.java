package deepwater.backends.tensorflow.test;

import com.sun.scenario.effect.ImageData;
import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.tensorflow.TensorflowBackend;
import deepwater.datasets.CIFAR10ImageDataset;
import deepwater.datasets.ImageDataSet;
import deepwater.datasets.MNISTImageDataset;
import deepwater.datasets.Pair;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

public class BackendInterfaceTest {

    @Test
    public void backendCanTrainMXNet() throws IOException {
        BackendTrain backend = new TensorflowBackend();

        MNISTImageDataset dataset = new MNISTImageDataset("/home/fmilo/workspace/h2o-3/train-labels-idx1-ubyte.gz",
                "/home/fmilo/workspace/h2o-3/train-images-idx3-ubyte.gz");

        List<Pair<Integer, float[]>> images = dataset.loadDigitImages();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();

        BackendModel model = backend.buildNet(dataset, opts, params,
                                            dataset.getNumClasses(), "lenet");

        for( Pair<Integer, float[]> entry: images ) {
            float[] image = entry.getSecond();
            Integer label = entry.getFirst();
            backend.train(model, image, new float[]{ label } );
            break;
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
           backend.train(model, image, new float[]{ label } );
           break;
       }
    }
}
