import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;

public class image_classify_test {

    static {
        System.loadLibrary("cudart");
        System.loadLibrary("cublas");
        System.loadLibrary("curand");
        System.loadLibrary("Native");
    }

    public static float[] img2pixels(String fname) throws IOException {
        int w = 224, h = 224;

        BufferedImage img = ImageIO.read(new File(fname.trim()));

        BufferedImage scaledImg = new BufferedImage(w, h, img.getType());

        Graphics2D g2d = scaledImg.createGraphics();
        g2d.drawImage(img, 0, 0, w, h, null);
        g2d.dispose();

        float[] pixels = new float[w * h * 3];

        int r_idx = 0;
        int g_idx = r_idx + w * h;
        int b_idx = g_idx + w * h;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                Color mycolor = new Color(scaledImg.getRGB(j, i));
                int red = mycolor.getRed();
                int green = mycolor.getGreen();
                int blue = mycolor.getBlue();
                pixels[r_idx] = red;
                r_idx++;
                pixels[g_idx] = green;
                g_idx++;
                pixels[b_idx] = blue;
                b_idx++;
            }
        }

        return pixels;
    }

    public static void main(String[] args) throws IOException {
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

        int batch_size = 40, start_index = 0, val_num = label_lst.size();

        int img_size = 224 * 224;

        int max_iter = 10;

        ImageClassify m = new ImageClassify();

        m.buildNet(10, batch_size, new String("/home/ops/Inception/Inception_BN-0039.params"));

        for (int iter = 0; iter < max_iter; iter++) {
            start_index = 0;
            long startTime = System.currentTimeMillis();

            while (start_index < val_num) {
                if (start_index + batch_size > val_num) {
                    start_index = val_num - batch_size;
                }

                float[] labels = new float[batch_size];
                float[] data = new float[batch_size * img_size * 3];

                for (int i = start_index; i < start_index + batch_size; i++) {
                    labels[i - start_index] = label_lst.get(i);
                    float[] tmp = img2pixels(img_lst.get(i));
                    for (int j = 0; j < img_size * 3; j++) {
                        data[(i - start_index) * img_size * 3 + j] = tmp[j];
                    }
                }

                float[] pred = m.train(data, labels);

                for (int i = 0; i < batch_size; i++) {
                    System.out.print((int)pred[i] + " ");
                }
                System.out.println();

                start_index = start_index + batch_size;
            }

            long endTime = System.currentTimeMillis();

            System.out.println("That took " + (endTime - startTime) + " milliseconds");
        }

    }

}
