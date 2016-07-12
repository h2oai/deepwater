package water.gpu;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by ops on 7/12/16.
 */
public final class util {
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
}
