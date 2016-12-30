package deepwater.backends.tensorflow.test;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendAPI;
import deepwater.backends.tensorflow.Client;
import deepwater.backends.tensorflow.ModelParams;
import deepwater.backends.tensorflow.TensorflowBackend;
import deepwater.datasets.BatchIterator;
import deepwater.datasets.ImageBatch;
import deepwater.datasets.MNISTImageDataset;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static deepwater.datasets.FileUtils.findFile;

public class TestGRPC {

    private Client client;

    @Before
    public void setUp(){
        client = new Client("localhost", 50051);
    }

    @After
    public void tearDown() throws InterruptedException {
        client.shutdown();
    }

    @Test
    public void testSimpleConnection() throws InterruptedException {
        client.Ping();
    }

    @Test
    public void testSimpleBuildNetwork() throws Exception {

        TensorflowBackend backend = new TensorflowBackend();
        ModelParams params = new ModelParams();
        params.put("height", 28);
        params.put("width", 28);
        params.put("channels", 1);
        params.put("classes", 10);
        BackendModel model = backend.buildModel("lenet", params);
        testMNIST(model);
    }

    public float testMNIST(BackendModel model) throws IOException {
        BackendAPI backend = new TensorflowBackend();
        MNISTImageDataset dataset = new MNISTImageDataset();
        String[] images = new String[]{
                findFile("bigdata/laptop/mnist/t10k-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/t10k-labels-idx1-ubyte.gz")
        };

        BatchIterator it = new BatchIterator(dataset, 1, images);
        ImageBatch b = new ImageBatch(dataset, 1024);

        while(it.next(b)){
            float[] loss = backend.predict(model, b.getImages(), b.getLabels());

            return loss[0];
        }


        return 0;
    }

}
