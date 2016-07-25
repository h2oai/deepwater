package h2o.deepwater.test;

import javax.imageio.ImageIO;

import water.gpu.ImageIter;
import water.gpu.ImagePred;
import water.gpu.ImageTrain;
import water.gpu.util;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class InceptionCLI {
    // load the cuda lib in CUDA_PATH, optional. theoretically we can find them if they are in LD_LIBRARY_PATH
    static {
        final boolean GPU = false;

        try {
            if (GPU) util.loadCudaLib();

            util.loadNativeLib("mxnet");
            util.loadNativeLib("Native");
        } catch (Exception e) {
            e.printStackTrace(System.out);
        }
    }

    static float[] loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(expandPath(path)));

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

    public static void main(String[] args) {
        ImagePred m = new ImagePred();

        // the path to Inception model
        m.setModelPath(expandPath(args[0]));

        m.loadInception();
        try {
            float[] pixels = loadImage(args[1]);
            System.out.println("\n\n" + m.predict(pixels) + "\n\n");
        } catch (Exception e) {
            e.printStackTrace(System.out);
        }
    }

    static String expandPath(String path) {
        return path.replaceFirst("^~", System.getProperty("user.home"));
    }

}

