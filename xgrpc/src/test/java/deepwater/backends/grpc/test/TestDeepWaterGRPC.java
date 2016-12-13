package deepwater.backends.grpc.test;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.RuntimeOptions;
import deepwater.backends.grpc.Client;
import deepwater.backends.grpc.GRPCBackendTrain;
import deepwater.datasets.BatchIterator;
import deepwater.datasets.ImageBatch;
import deepwater.datasets.MNISTImageDataset;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import deepwater.utils.PythonWorkerPool;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import static deepwater.datasets.FileUtils.findFile;

public class TestDeepWaterGRPC {

    private static PythonWorkerPool pypool;

    private Client client;
    private GRPCBackendTrain backend;

    @BeforeClass
    public static void startPythonDaemon(){
        pypool = new PythonWorkerPool();
    }

    @AfterClass
    public static void stopPythonDaemon(){
        pypool.stop();
    }

    @Before
    public void setUp() {
        HashMap<String, String> env = new HashMap<>();
        String userHome = System.getProperty("user.home");
        env.put("PYTHONPATH", userHome + "/workspace/deepwater/xgrpc/src/main/python/");
        env.put("LD_LIBRARY_PATH", "/usr/local/cuda/lib64");
        pypool.createPythonWorker(userHome + "/anaconda3/envs/deepwater/bin/python", env);
        client = new Client("localhost", 50051);

        backend = new GRPCBackendTrain("localhost", 50051);

    }

    @After
    public void tearDown() throws InterruptedException {
        client.shutdown();
    }

    @Test
    public void testSimpleBuildNetwork() throws Exception {

        String name = "mlp";
        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams backend_params = new BackendParams();
        int num_classes = dataset.getNumClasses();

        BackendModel model = backend.buildNet(dataset, opts, backend_params, num_classes, name);

        String[] images = new String[]{
                findFile("bigdata/laptop/mnist/t10k-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/t10k-labels-idx1-ubyte.gz")
        };

        BatchIterator it = new BatchIterator(dataset, 1, images);
        ImageBatch b = new ImageBatch(dataset, 10);

        // can train
        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
                break;
            }
            testMNIST(model);
        }

        // can predict
        while(it.next(b)){
            backend.predict(model, b.getImages(), b.getLabels());
            break;
        }

        backend.delete(model);
    }


    private void backendCanSaveCheckpoint(String modelName, int batchSize) throws IOException {

        String[] train_images = new String[]{
                findFile("bigdata/laptop/mnist/train-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/train-labels-idx1-ubyte.gz"),
        };
        MNISTImageDataset dataset = new MNISTImageDataset();

        RuntimeOptions opts = new RuntimeOptions();
        BackendParams params = new BackendParams();
        BackendModel model = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelName);

        float initial = testMNIST(model);

        BatchIterator it = new BatchIterator(dataset, 1, train_images);
        ImageBatch b = new ImageBatch(dataset, batchSize);
        while(it.nextEpochs()) {
            while (it.next(b)) {
                backend.train(model, b.getImages(), b.getLabels());
            }
        }
        float trained = testMNIST(model);

        //create a temp file
        File modelFile = File.createTempFile("model", ".tmp");
        backend.saveModel(model, modelFile.getAbsolutePath());

        File modelParams = File.createTempFile("params", ".tmp");
        backend.saveParam(model, modelParams.getAbsolutePath());

        BackendModel model2 = backend.buildNet(dataset, opts, params, dataset.getNumClasses(), modelFile.getAbsolutePath());

        backend.loadParam(model2, modelParams.getAbsolutePath());

        float loaded = testMNIST(model2);

        assert initial < trained: initial;
        assert trained <= loaded: loaded;

        backend.delete(model);
        backend.delete(model2);
    }

    @Test
    public void testMLP() throws IOException{
        backendCanSaveCheckpoint("mlp", 10);
    }

    private float testMNIST(BackendModel model) throws IOException {

        MNISTImageDataset dataset = new MNISTImageDataset();
        String[] images = new String[]{
                findFile("bigdata/laptop/mnist/t10k-images-idx3-ubyte.gz"),
                findFile("bigdata/laptop/mnist/t10k-labels-idx1-ubyte.gz")
        };

        BatchIterator it = new BatchIterator(dataset, 1, images);
        ImageBatch b = new ImageBatch(dataset, 10);

        while(it.next(b)){
            float[] loss = backend.predict(model, b.getImages(), b.getLabels());
            break;
        }


        return 0;
    }
}
