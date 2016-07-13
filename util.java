package water.gpu;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

import static java.nio.file.FileVisitResult.CONTINUE;
import static java.nio.file.FileVisitResult.TERMINATE;

public final class util {

    public static void deleteFileOrFolder(final Path path) throws IOException {
        Files.walkFileTree(path, new SimpleFileVisitor<Path>(){
            @Override public FileVisitResult visitFile(final Path file, final BasicFileAttributes attrs)
                    throws IOException {
                Files.delete(file);
                return CONTINUE;
            }

            @Override public FileVisitResult visitFileFailed(final Path file, final IOException e) {
                return handleException(e);
            }

            private FileVisitResult handleException(final IOException e) {
                e.printStackTrace(); // replace with more robust error handling
                return TERMINATE;
            }

            @Override public FileVisitResult postVisitDirectory(final Path dir, final IOException e)
                    throws IOException {
                if(e!=null)return handleException(e);
                Files.delete(dir);
                return CONTINUE;
            }
        });
    };

    public static float[] readFC(String fname, int length) throws IOException {
        float[] res = new float[length];
        FileChannel inChannel = new RandomAccessFile(fname, "rw").getChannel();
        ByteBuffer buffer = ByteBuffer.allocate(Float.BYTES * res.length);
        inChannel.read(buffer);
        buffer.flip();
        FloatBuffer buffer2 = buffer.asFloatBuffer();
        for (int i = 0; i < res.length; i++) {
            res[i] = buffer2.get(i);
        }
        inChannel.close();
        return res;
    }

    public static void writeFC(String fname, float[] res) throws IOException{
        new File(fname).createNewFile();
        ByteBuffer bbuffer = ByteBuffer.allocate(4 * res.length);
        FloatBuffer buffer = bbuffer.asFloatBuffer();
        for (int i = 0; i < res.length; i++) buffer.put(res[i]);
        buffer.flip();
        FileChannel fc = new RandomAccessFile(fname, "rw").getChannel();
        fc.write(bbuffer);
        fc.close();
    }

    public static float[] img2pixels(String fname, int w, int h, String ftmp) throws IOException {

        float[] pixels = new float[w * h * 3];

        if (new File(ftmp).exists()) {
            pixels = readFC(ftmp, pixels.length);
        } else {
            // resize the image
            BufferedImage img = ImageIO.read(new File(fname.trim()));
            BufferedImage scaledImg = new BufferedImage(w, h, img.getType());
            Graphics2D g2d = scaledImg.createGraphics();
            g2d.drawImage(img, 0, 0, w, h, null);
            g2d.dispose();

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
            writeFC(ftmp, pixels);
        }
        return pixels;
    }
}
