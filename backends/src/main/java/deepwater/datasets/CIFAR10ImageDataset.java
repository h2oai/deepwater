package deepwater.datasets;


import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class CIFAR10ImageDataset extends ImageDataSet {

    public CIFAR10ImageDataset(){
        super(32, 32, 3, 10);
    }

    public List<Pair<Integer,float[]>> loadImages(String filepath) throws IOException {
        List<Pair<Integer,float[]>> images = new ArrayList<>();

        InputStream inputStream = new FileInputStream(filepath);

        int read = 0;
        byte[] buffer = new byte[1+32*32*3];

        while((read = inputStream.read(buffer, 0, buffer.length)) != -1) {

            int label = buffer[0] % 0xFF;
            float[] imageDataFloat = new float[32*32*3];
            int i = 0;
            for (int j = 1; j < imageDataFloat.length; j++) {
               float result = buffer[j] & 0xFF;
               imageDataFloat[i++] = result;
            }

            images.add(new Pair<>(label, imageDataFloat));
        }

        return images;
    }
}
