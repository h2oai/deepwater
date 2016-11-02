package deepwater.examples;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.caffe.CaffeBackend;

public class ImageNet {
    public static void main(String[] args) throws Exception {
        CaffeBackend caffe = new CaffeBackend();
        BackendParams params = new BackendParams();
        params.set("graph", "src/main/java/deepwater/examples/googlenet.prototxt");
        final int batch = 32;
        ImageNetDataset ds = new ImageNetDataset("../../caffe/examples/imagenet/ilsvrc12_train_lmdb", batch);
        BackendModel model = caffe.buildNet(ds, null, params, 1000, "google");

        float[] data = new float[batch * ds.getChannels() * ds.getHeight() * ds.getWidth()];
        float[] labs = new float[batch];
        for (int i = 0; i < 100; i++) {
            ds.batch(data, labs);
            caffe.train(model, data, labs);
        }

        System.out.println("Saving model");
        caffe.saveParam(model, "google_snapshot");
        System.out.println("Done");
    }
}
