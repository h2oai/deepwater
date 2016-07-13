package water.gpu;

import java.io.File;
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;

public class imagetest {

    static {
        System.loadLibrary("cudart");
        System.loadLibrary("cublas");
        System.loadLibrary("curand");
        System.loadLibrary("Native");
    }

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(new File("/home/ops/Desktop/sf1_train.lst")));

        System.out.println("Integer.BYTES " + Integer.BYTES);
        System.out.println("Float.BYTES " + Float.BYTES);
        System.out.println("Double.BYTES " + Double.BYTES);

        String line = null;

        ArrayList<Float> label_lst = new ArrayList<>();
        ArrayList<String> img_lst = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            String[] tmp = line.split("\t");

            label_lst.add(new Float(tmp[1]).floatValue());
            img_lst.add(tmp[2]);
        }

        br.close();

        int batch_size = 40;//, start_index = 0, val_num = label_lst.size();

        int img_size = 224 * 224;

        int max_iter = 10;

        ImageClassify m = new ImageClassify();

        m.buildNet(10, batch_size);

        ImageIter img_iter = new ImageIter(img_lst, label_lst, batch_size, "/tmp", 224, 224);

        for (int iter = 0; iter < max_iter; iter++) {
            img_iter.Reset();
            while(img_iter.Nest()){
                float[] data = img_iter.getData();
                float[] labels = img_iter.getLabel();
                float[] pred = m.train(data, labels, true);

                int count = 0;
                for (int i = 0; i < batch_size; i++) {
                    System.out.print((int)pred[i] + " ");
                    if (pred[i] == labels[i]) count++;
                }
                System.out.println("Acc: " + (double)count / batch_size);
                for (int i = 0; i < batch_size; i++) {
                    System.out.print((int)labels[i] + " ");
                }
                System.out.println();
            }
            img_iter.Reset();
            ArrayList<Float> train_pred = new ArrayList<>();
            while(img_iter.Nest()){
                float[] data = img_iter.getData();
                float[] labels = img_iter.getLabel();
                float[] pred = m.train(data, labels, false);
                for (int i = 0; i < batch_size; i++) {
                    train_pred.add(pred[i]);
                }
            }
            int count = 0;
            for (int i = 0; i < label_lst.size(); i++) {
                if (train_pred.get(i).equals(label_lst.get(i))) count++;
            }
            System.out.println("Iter: " + iter + " Acc: " + (double)count / label_lst.size());
        }

    }

}
