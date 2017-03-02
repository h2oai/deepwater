package deepwater.backends.tensorflow.test.datasets;

import deepwater.datasets.FileUtils;
import deepwater.datasets.ImageDataSet;
import deepwater.datasets.Pair;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;

public class CatDogMouseImageDataset extends ImageDataSet {

    public CatDogMouseImageDataset(){
        super(224, 224, 3, 3);
    }

    private Map<String, Integer> labels = new HashMap<String, Integer>() {{
        this.put("cat", 0);
        this.put("dog", 1);
        this.put("mouse", 2);
    }};

    public List<Pair<Integer,float[]>> loadImages(String... filepath) throws IOException {
        List<Pair<Integer,float[]>> images = new ArrayList<>();

        for (String path: filepath) {
            Scanner scanner = new Scanner(new File(path));

            while (scanner.hasNextLine()) {
                String[] pathLabel = scanner.nextLine().split("\\s");
                File img = new File(FileUtils.findFile(pathLabel[0]));

                // RESIZE
                InputStream inputStream = new FileInputStream(img);
                byte[] imgBytes = new byte[(int)img.length()];

                inputStream.read(imgBytes);

                BufferedImage inputImage = ImageIO.read(img);

                // READ RESIZED
                BufferedImage scaledImg = new BufferedImage(224, 224, inputImage.getType());
                Graphics2D g2d = scaledImg.createGraphics();
                g2d.drawImage(inputImage, 0, 0, 224, 224, null);
                g2d.dispose();

                int r_idx = 0;
                int g_idx = r_idx + 224 * 224;
                int b_idx = g_idx + 224 * 224;

                float[] pixels = new float[224*224*3];

                for (int i = 0; i < 224; i++) {
                    for (int j = 0; j < 224; j++) {
                        Color mycolor = new Color(scaledImg.getRGB(j, i));
                        int red = mycolor.getRed();
                        int green = mycolor.getGreen();
                        int blue = mycolor.getBlue();
                        pixels[r_idx] = red;
                        pixels[g_idx] = green;
                        pixels[b_idx] = blue;
                        r_idx++;
                        g_idx++;
                        b_idx++;
                    }
                }

                images.add(new Pair<>(labels.get(pathLabel[1]), pixels));
            }
        }

        return images;
    }
}
