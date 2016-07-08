package water.gpu;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class imagetest {

    static {
        System.loadLibrary("Native");
    }

    public static void main(String[] args) throws IOException {
        // write your code here
        BufferedImage img = ImageIO.read(new File("/home/ops/Documents/h2o-native/test1.jpg"));

        int w = 224, h = 224;

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
                pixels[r_idx] = red; r_idx++;
                pixels[g_idx] = green; g_idx++;
                pixels[b_idx] = blue; b_idx++;
            }
        }

        ImageNative m = new ImageNative();

        m.setModelPath("/home/ops/Desktop/kaggle_statefarm/inception/model");

        m.loadInception();

        System.out.println("Prediction: " + m.predict(pixels));

    }

}
