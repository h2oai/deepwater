package deepwater.datasets;


import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class CIFAR10ImageDataset extends ImageDataSet {

    public CIFAR10ImageDataset(){
        super(32, 32, 3, 10);
    }

    public List<Pair<Integer,float[]>> loadImages(String... filepath) throws IOException {
        List<Pair<Integer,float[]>> images = new ArrayList<>();

        InputStream inputStream = new FileInputStream(filepath[0]);

        // Read one image
        byte[] buffer = new byte[1+(32*32*3)];

        // Convert image to [w, h, channels]
        while((inputStream.read(buffer, 0, buffer.length)) != -1) {

            int label = buffer[0] % 0xFF;
            float[][] imageDataFloat = new float[3][32*32];

            int i = 1;
            for (int channel = 0; channel < 3; channel++) {
                for (int j = 0; j < 32*32; j++) {
                    float result = buffer[i++] & 0xFF;
                    imageDataFloat[channel][j] = result;
                }
            }
            assert i == 1+32*32*3;

            float[] result = new float[32*32*3];
            int k=0;
            for (int j=0; j < 32*32; j++) {
                for (int channel = 0; channel < 3; channel++) {
                    result[k++] = imageDataFloat[channel][j];
                }
            }

            assert k == 32*32*3;

//            BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
//            WritableRaster raster = (WritableRaster) image.getRaster();
//            raster.setPixels(0, 0, 32, 32, result);
//
//            ImageIcon icon = new ImageIcon(image);
//            JLabel jlabel = new JLabel(icon);
//            JOptionPane.showMessageDialog(null, jlabel);


            images.add(new Pair<>(label, result));
        }

        return images;
    }
}
