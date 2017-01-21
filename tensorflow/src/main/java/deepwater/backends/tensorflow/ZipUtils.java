package deepwater.backends.tensorflow;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

public class ZipUtils {

  public static File[] extractFiles(String zipFile, String extractFolder) {
    List<File> extractedFiles = new LinkedList();
    try {
      int BUFFER = 2048;
      File file = new File(zipFile);
      ZipFile zip = new ZipFile(new File(zipFile));
      String newPath = extractFolder;

      new File(newPath).mkdir();
      Enumeration zipFileEntries = zip.entries();

      // Process each entry
      while (zipFileEntries.hasMoreElements()) {
        // grab a zip file entry
        ZipEntry entry = (ZipEntry) zipFileEntries.nextElement();
        String currentEntry = entry.getName();

        File destFile = new File(newPath, currentEntry);
        File destinationParent = destFile.getParentFile();

        // create the parent directory structure if needed
        destinationParent.mkdirs();

        if (!entry.isDirectory()) {
          BufferedInputStream is = new BufferedInputStream(zip
                  .getInputStream(entry));
          int currentByte;
          // establish buffer for writing file
          byte data[] = new byte[BUFFER];

          // write the current file to disk
          System.out.println("extracting to "+destFile.getAbsolutePath());
          extractedFiles.add(destFile);

          FileOutputStream fos = new FileOutputStream(destFile);
          BufferedOutputStream dest = new BufferedOutputStream(fos,
                  BUFFER);

          // read and write until last byte is encountered
          while ((currentByte = is.read(data, 0, BUFFER)) != -1) {
            dest.write(data, 0, currentByte);
          }
          dest.flush();
          dest.close();
          is.close();
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
    File[] result = new File[extractedFiles.size()];
    extractedFiles.toArray(result);
    return result;
  }

  public static void zipFiles(File zipFile, File[] entries) {
    byte[] buffer = new byte[2048];
    try (
      FileOutputStream fos = new FileOutputStream(zipFile);
      ZipOutputStream zos = new ZipOutputStream(fos);
    ){
      for (File file : entries) {
        ZipEntry ze = new ZipEntry( file.getName());
        zos.putNextEntry(ze);
        // Copy File Content
        try(FileInputStream in = new FileInputStream(file))
        {
          int len;
          while ((len = in.read(buffer)) > 0)
          {
            zos.write(buffer, 0, len);
          }
        }
        zos.closeEntry();
      }
    } catch (IOException ex) {
      ex.printStackTrace();
    }
  }

}
