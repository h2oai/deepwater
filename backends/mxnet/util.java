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
import java.nio.file.Paths;

import static java.nio.file.FileVisitResult.CONTINUE;
import static java.nio.file.FileVisitResult.TERMINATE;

public final class util {

    public static String path(String dirname, String ... more ){
        return Paths.get(dirname, more).toString();
    }

    public static String libName(String name){
        String OS = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
        
        String lib_suffix = "";

        if ((OS.indexOf("mac") >= 0) || (OS.indexOf("darwin") >= 0)) {
            lib_suffix = "so";
        } else if (OS.indexOf("nux") >= 0) {
            lib_suffix = "so";
        } else {
            System.err.println("Operating system not supported: "+OS);
        }

        return "lib" + name + "." + lib_suffix;
    }

    public static String getNvidiaStats() throws java.io.IOException {
      String cmd = "nvidia-smi";
      InputStream stdin = Runtime.getRuntime().exec(cmd).getInputStream();
      InputStreamReader isr = new InputStreamReader(stdin);
      BufferedReader br = new BufferedReader(isr);
      StringBuilder sb = new StringBuilder();
      String s = null;
      while ((s = br.readLine()) != null) {
        sb.append(s + "\n");
      }
      return sb.toString();
    }

    public static void loadCudaLib() {
        String cuda_path = System.getenv().get("CUDA_PATH");
        checkNotNull(cuda_path,"CUDA_PATH hasn't been set!");

        System.load(path(cuda_path, "lib64", libName("cudart")));
        System.load(path(cuda_path, "lib64", libName("cublas")));
        System.load(path(cuda_path, "lib64", libName("curand")));
        System.load(path(cuda_path, "lib64", libName("cudnn")));
    }

    public static String extractLibrary(String resourceName) throws IOException {

        String libname = libName(resourceName);
        String origin = path("/water/gpu/",libname);

        String tmpdir = System.getProperty("java.io.tmpdir");
        if (tmpdir.isEmpty()){
            tmpdir = "/tmp";
        }
        String target = path(tmpdir,libname);
        if (Files.exists(Paths.get(target))) {
            Files.delete(Paths.get(target));
        }

        InputStream in = util.class.getResourceAsStream(origin);
        checkNotNull(in,"No native lib " + origin + " found in jar. Please check installation!");

        OutputStream out = new FileOutputStream(target);
        checkNotNull(out,"could not create file");
        copy(in, out);
        return target;
    }

    public static void loadNativeLib(String resourceName) throws IOException {
        System.load(extractLibrary(resourceName));
    }

    public static <T> T checkNotNull(T reference, String msg) {
   if (reference == null) {
     throw new NullPointerException(msg);
    }
    return reference;
  }

  private static final int BUF_SIZE = 0x1000; // 4K

  public static long copy(InputStream from, OutputStream to)
      throws IOException {
    byte[] buf = new byte[BUF_SIZE];
    long total = 0;
    while (true) {
      int r = from.read(buf);
      if (r == -1) {
        break;
      }
      to.write(buf, 0, r);
      total += r;
    }
    return total;
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
        ByteBuffer buffer = ByteBuffer.allocate(4 * res.length);
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
        img2pixels(fname,w,h,3,pixels,0,null);
        return pixels;
    }

    public static void img2pixels(String fname, int w, int h, int channels, float[] pixels, int start, float[] mean) throws IOException {
        // resize the image
        BufferedImage img = ImageIO.read(new File(fname.trim()));
        BufferedImage scaledImg = new BufferedImage(w, h, img.getType());
        Graphics2D g2d = scaledImg.createGraphics();
        g2d.drawImage(img, 0, 0, w, h, null);
        g2d.dispose();

        int r_idx = start;
        int g_idx = r_idx + w * h;
        int b_idx = g_idx + w * h;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                Color mycolor = new Color(scaledImg.getRGB(j, i));
                int red = mycolor.getRed();
                int green = mycolor.getGreen();
                int blue = mycolor.getBlue();
		if (channels==1) {
			pixels[r_idx] = (red+green+blue)/3;
                        if (mean!=null) {
                            pixels[r_idx] -= mean[r_idx];
                        }
		} else {
			pixels[r_idx] = red;
			pixels[g_idx] = green;
			pixels[b_idx] = blue;
                        if (mean!=null) {
                            pixels[r_idx] -= mean[r_idx-start];
                            pixels[g_idx] -= mean[g_idx-start];
                            pixels[b_idx] -= mean[b_idx-start];
                        }
		}
                r_idx++;
                g_idx++;
                b_idx++;
            }
        }
    }
}
