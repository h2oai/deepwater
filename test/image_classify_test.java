
import java.io.File;
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;

public class imagetest {

    public static void main(String[] args) throws IOException {

        util.loadCudaLib();
        util.loadNativeLib("mxnet");
        util.loadNativeLib("Native");

        // just read the path and label for each img
        BufferedReader br = new BufferedReader(new FileReader(new File("sf1_train.lst")));

        String line = null;

        ArrayList<Float> label_lst = new ArrayList<>();
        ArrayList<String> img_lst = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            String[] tmp = line.split("\t");

            label_lst.add(new Float(tmp[1]).floatValue());
            img_lst.add(tmp[2]);
        }

        br.close();

        int batch_size = 40;

        int max_iter = 10;

        ImageTrain m = new ImageTrain();

        m.buildNet(10, batch_size, "inception_bn");

        ImageIter img_iter = new ImageIter(img_lst, label_lst, batch_size, 224, 224);

        for (int iter = 0; iter < max_iter; iter++) {
            img_iter.Reset();
            while(img_iter.Nest()){
                float[] data = img_iter.getData();
                float[] labels = img_iter.getLabel();
                // the return values are probs
                float[] pred = m.train(data, labels);
            }
        }

    }

}
