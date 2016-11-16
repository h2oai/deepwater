package deepwater.datasets;


import java.io.File;

public class FileUtils {

    public static String expandPath(String path) {
        return path.replaceFirst("^~", System.getProperty("user.home"));
    }

    public static String findFile(String fname) {
        // When run from eclipse, the working directory is different.
        // Try pointing at another likely place

        File file = new File(fname);
        if (!file.exists())
            file = new File("target/" + fname);
        if (!file.exists())
            file = new File("../" + fname);
        if (!file.exists())
            file = new File("../../" + fname);
        if (!file.exists())
            file = new File("../target/" + fname);
        if (!file.exists())
            file = new File(expandPath(fname));
        if (!file.exists())
            file = null;

        if (file == null) {
            return "";
        }

        return file.getAbsolutePath();
    }

}
