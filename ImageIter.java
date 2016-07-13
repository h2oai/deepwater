package water.gpu;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.nio.file.Path;

import static water.gpu.util.deleteFileOrFolder;
import static water.gpu.util.img2pixels;

public class ImageIter {

    public ImageIter(ArrayList<String> img_lst, ArrayList<Float> lable_lst, int batch_size, String tmp_path, int width, int height) throws IOException{
        assert img_lst.size() == lable_lst.size();
        this.img_lst = img_lst;
        this.label_lst = lable_lst;
        this.batch_size = batch_size;
        this.val_num = img_lst.size();
        start_index = 0;
        this.width = width;
        this.height = height;
        this.tmp_path = tmp_path + File.separator + "ImageIter";
        data = new float[batch_size * width * height * 3];
        label = new float[batch_size];

        if (new File(this.tmp_path).exists()) {
            deleteFileOrFolder(new File(this.tmp_path).toPath());
        } else {
            new File(this.tmp_path).mkdir();
        }

    }

    public void Reset() {
        start_index = 0;
    }

    public boolean Nest() throws IOException {
        if (start_index < val_num) {
            if (start_index + batch_size > val_num) {
                start_index = val_num - batch_size;
            }

            System.out.println("start_index " + start_index);

            for (int i = start_index; i < start_index + batch_size; i++) {
                label[i - start_index] = label_lst.get(i) ;
                float[] tmp = img2pixels(img_lst.get(i), width, height, this.tmp_path + File.separator + i + ".out");
                for (int j = 0; j < width * height * 3; j++) {
                    data[(i - start_index) * width * height * 3 + j] = tmp[j];
                }
            }

            start_index = start_index + batch_size;

            return true;
        } else {
            return false;
        }
    }

    public float[] getData() { return data;}

    public float[] getLabel() { return label;}

    private int val_num;
    private int start_index;
    private int batch_size;
    private int width, height;
    private String tmp_path;
    private float[] data;
    private float[] label;
    private ArrayList<String> img_lst;
    private ArrayList<Float> label_lst;

    public static void main(String[] args) throws IOException {

    }
}
