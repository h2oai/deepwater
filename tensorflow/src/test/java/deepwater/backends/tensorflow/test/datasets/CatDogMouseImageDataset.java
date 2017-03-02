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

        File temp = new File("/tmp/test/");
        temp.deleteOnExit();
        temp.mkdir();

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

                // creates output image
                BufferedImage outputImage = new BufferedImage(224, 224, inputImage.getType());
                // scales the input image to the output image
                Graphics2D g2d = outputImage.createGraphics();
                g2d.drawImage(inputImage, 0, 0, 224, 224, null);
                g2d.dispose();

                ImageIO.write(outputImage, "jpg", new File("/tmp/test/" + img.getName()));

                // READ RESIZED
                img = new File("/tmp/test/"+img.getName());
                inputStream = new FileInputStream(img);
                imgBytes = new byte[(int)img.length()];

                inputStream.read(imgBytes);

                int size = imgBytes.length / 3;
                float[][] imageDataFloat = new float[3][size];

                int i = 0;
                for (int channel = 0; channel < 3; channel++) {
                    for (int j = 0; j < size; j++) {
                        float result = imgBytes[i++] & 0xFF;
                        imageDataFloat[channel][j] = result;
                    }
                }

                float[] result = new float[imgBytes.length];
                int k = 0;
                for (int j = 0; j < size; j++) {
                    for (int channel = 0; channel < 3; channel++) {
                        result[k++] = imageDataFloat[channel][j];
                    }
                }

                images.add(new Pair<>(labels.get(pathLabel[1]), result));
            }
        }

        return images;
    }
}
