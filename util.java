package water.gpu;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Map;
import java.util.Locale;

import static java.nio.file.FileVisitResult.CONTINUE;
import static java.nio.file.FileVisitResult.TERMINATE;

public final class util {

    public static boolean loadCudaLib() {
        Map<String, String> env = System.getenv();
        String cuda_path = env.get("CUDA_PATH");
        if (cuda_path == null) {
            System.err.println("CUDA_PATH hasn't been set!");
            return false;
        }

        String OS = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);

        String lib_suffix = "";

        if ((OS.indexOf("mac") >= 0) || (OS.indexOf("darwin") >= 0)) {
            lib_suffix = "dylib";
        } else if (OS.indexOf("nux") >= 0) {
            lib_suffix = "so";
        } else {
            System.err.println("Not support Operating system!");
        }

        // load related cuda libraries in case the system can't find them automatically
        System.load(cuda_path + "lib64" + File.separator + "libcudart." + lib_suffix);
        System.load(cuda_path + "lib64" + File.separator + "libcublas." + lib_suffix);
        System.load(cuda_path + "lib64" + File.separator + "libcurand." + lib_suffix);
        return true;
    }

    public static boolean loadNativeLib(String resourceName) throws IOException {
        String OS = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
        String lib_suffix = "";

        if ((OS.indexOf("mac") >= 0) || (OS.indexOf("darwin") >= 0)) {
            lib_suffix = "dylib";
        } else if (OS.indexOf("nux") >= 0) {
            lib_suffix = "so";
        } else {
            System.err.println("Not support Operating system!");
        }

        InputStream stream = util.class.getResourceAsStream("/water/gpu/lib" + resourceName + "." + lib_suffix);
        if (stream == null) {
            System.err.println("No native libs found in jar. Please check installation!");
            return false;
        }

        int readBytes;
        byte[] buffer = new byte[4096];

        OutputStream resStreamOut = new FileOutputStream("/tmp" + File.separator + "lib" + resourceName + "." + lib_suffix);
        while ((readBytes = stream.read(buffer)) > 0) {
            resStreamOut.write(buffer, 0, readBytes);
        }

        System.load("/tmp" + File.separator + "lib" + resourceName + "." + lib_suffix);

        return true;
    }

    public static void deleteFileOrFolder(final Path path) throws IOException {
        Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(final Path file, final BasicFileAttributes attrs)
                    throws IOException {
                Files.delete(file);
                return CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(final Path file, final IOException e) {
                return handleException(e);
            }

            private FileVisitResult handleException(final IOException e) {
                e.printStackTrace(); // replace with more robust error handling
                return TERMINATE;
            }

            @Override
            public FileVisitResult postVisitDirectory(final Path dir, final IOException e)
                    throws IOException {
                if (e != null) return handleException(e);
                Files.delete(dir);
                return CONTINUE;
            }
        });
    }

    ;

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

    public static void writeFC(String fname, float[] res) throws IOException {
        new File(fname).createNewFile();
        ByteBuffer bbuffer = ByteBuffer.allocate(4 * res.length);
        FloatBuffer buffer = bbuffer.asFloatBuffer();
        for (int i = 0; i < res.length; i++) buffer.put(res[i]);
        buffer.flip();
        FileChannel fc = new RandomAccessFile(fname, "rw").getChannel();
        fc.write(bbuffer);
        fc.close();
    }

    public static float[] img2pixels(String fname, int w, int h) throws IOException {

        float[] pixels = new float[w * h * 3];
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
        return pixels;
    }
}
