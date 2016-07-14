
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

        BufferedReader br = new BufferedReader(new FileReader(new File("/home/ops/Desktop/sf1_train.lst")));

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

        ImageClassify m = new ImageClassify();

        m.buildNet(10, batch_size);

        ImageIter img_iter = new ImageIter(img_lst, label_lst, batch_size, 224, 224);

        for (int iter = 0; iter < max_iter; iter++) {
            img_iter.Reset();
            while(img_iter.Nest()){
                float[] data = img_iter.getData();
                float[] labels = img_iter.getLabel();
                float[] pred = m.train(data, labels, true);
                System.out.println("pred " + pred.length);

                for (int i = 0; i < batch_size * 10; i++) {
                    System.out.print(pred[i] + " ");
                }
                System.out.println();
            }
        }

    }

}
