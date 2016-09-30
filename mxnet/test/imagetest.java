
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

import water.gpu.ImagePred;
import water.gpu.util;

public class imagetest {

  public static void main(String[] args) throws IOException {
    
    // load the cuda lib in CUDA_PATH, optional.
    // theorecticlly we can find them if they are in LD_LIBRARY_PATH
    // util.loadCudaLib();
    // load the native code;
    util.loadNativeLib("mxnet");
    util.loadNativeLib("Native");
    // an image for testing
    BufferedImage img = ImageIO.read(new File("test1.jpg"));

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

    ImagePred m = new ImagePred();

    // the path to Inception model
    m.setModelPath("Inception");

    m.loadInception();

    System.out.println("Prediction: " + m.predict(pixels));

  }
}
