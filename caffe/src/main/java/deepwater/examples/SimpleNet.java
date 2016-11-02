package deepwater.examples;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.caffe.CaffeBackend;

public class SimpleNet {
    public static void main(String[] args) throws Exception {
        CaffeBackend caffe = new CaffeBackend();
        BackendParams params = new BackendParams();
        params.set("graph", "src/main/java/deepwater/examples/simplenet.prototxt");
        final int batch = 32;
        MNISTDataset ds = new MNISTDataset("../../caffe/examples/mnist/mnist_train_lmdb", batch);
        BackendModel model = caffe.buildNet(ds, null, params, 10, "simple");

        float[] data = new float[batch * ds.getChannels() * ds.getHeight() * ds.getWidth()];
        float[] labs = new float[batch];
        for (int i = 0; i < 100000; i++) {
            ds.batch(data, labs);
            caffe.train(model, data, labs);
        }

        System.out.println("Saving model");
        caffe.saveParam(model, "simple_snapshot");
        System.out.println("Done");
    }
}
