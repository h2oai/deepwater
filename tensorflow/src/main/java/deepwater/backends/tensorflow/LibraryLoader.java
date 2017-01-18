package deepwater.backends.tensorflow;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;

public final class LibraryLoader {

    public static String path(String dirname, String ... more ){
        return Paths.get(dirname, more).toString();
    }

    /**
     * http://stackoverflow.com/questions/15409223/adding-new-paths-for-native-libraries-at-runtime-in-java
     * Adds the specified path to the java library path
     *
     * @param pathToAdd the path to add
     * @throws Exception
     */
    public static void addLibraryPath(String pathToAdd) throws Exception{
        final Field usrPathsField = ClassLoader.class.getDeclaredField("usr_paths");
        usrPathsField.setAccessible(true);

        //get array of paths
        final String[] paths = (String[])usrPathsField.get(null);

        //check if the path to add is already present
        for(String path : paths) {
            if(path.equals(pathToAdd)) {
                return;
            }
        }

        //add the new path
        final String[] newPaths = Arrays.copyOf(paths, paths.length + 1);
        newPaths[newPaths.length-1] = pathToAdd;
        usrPathsField.set(null, newPaths);
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

    public static String extractLibrary(String resourceName) throws IOException {

        String libname = libName(resourceName);

        String tmpdir = System.getProperty("java.io.tmpdir");
        if (tmpdir.isEmpty()){
            tmpdir = "/tmp";
        }
        String target = path(tmpdir,libname);
        if (Files.exists(Paths.get(target))) {
            Files.delete(Paths.get(target));
        }

        InputStream in = LibraryLoader.class.getClassLoader().getResourceAsStream(libname);
        checkNotNull(in,"No native lib " + libname + " found in jar. Please check installation!");

        OutputStream out = new FileOutputStream(target);
        checkNotNull(out,"could not create file");
        try {
            addLibraryPath(tmpdir);
        } catch (Exception e) {
            e.printStackTrace();
        }
        copy(in, out);
        return target;
    }

    public static void loadNativeLib(String resourceName) throws IOException {
        extractLibrary(resourceName);
        System.loadLibrary(resourceName);
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

}
