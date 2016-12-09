package deepwater.backends.grpc.test;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
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

import java.util.HashMap;

import static deepwater.datasets.FileUtils.findFile;

public class TestDeepWaterGRPC {

    private static PythonWorkerPool pypool;

    private Client client;

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
        env.put("PYTHONPATH", "/home/fmilo/workspace/deepwater/xgrpc/src/main/python/");
        env.put("LD_LIBRARY_PATH", "/usr/local/cuda/lib64");
        pypool.createPythonWorker("/home/fmilo/anaconda2/bin/python", env);
        client = new Client("localhost", 50051);
    }

    @After
    public void tearDown() throws InterruptedException {
        client.shutdown();
    }

    @Test
    public void testSimpleBuildNetwork() throws Exception {

        BackendTrain backend = new GRPCBackendTrain("localhost", 50051);

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
        ImageBatch b = new ImageBatch(dataset, 1024);

        while(it.next(b)){
            backend.predict(model, b.getImages(), b.getLabels());
        }
    }

}
