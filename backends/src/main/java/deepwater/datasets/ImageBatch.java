package deepwater.datasets;

import java.util.Arrays;

public class ImageBatch {

    final int size;
    final float[] labels;
    final float[] images;

    public ImageBatch(ImageDataSet data, int size) {
        this.size = size;
        this.labels = new float[data.getNumClasses() * size];
        this.images = new float[data.getHeight() * data.getWidth() * data.getChannels() * size];
    }

    public void reset() {
        Arrays.fill(labels, 0);
        Arrays.fill(images, 0);
    }

    public float[] getLabels() {
        return labels;
    }

    public float[] getImages() {
        return images;
    }

}
