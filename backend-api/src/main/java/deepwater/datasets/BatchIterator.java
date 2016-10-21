package deepwater.datasets;

import java.io.IOException;
import java.util.List;

public class BatchIterator {

    private final ImageDataSet data;
    private final int totalEpochs;
    private int currentEpoch;
    private List<Pair<Integer, float[]>> imageLabelList;
    private int savedIterator;
    private String[] images;

    public BatchIterator(ImageDataSet data, int epochs, String... images) {
        this.data = data;
        this.totalEpochs = epochs;
        this.currentEpoch = 0;
        this.images = new String[]{};
        this.images = images;
    }

    public void newEpoch() throws IOException {
       this.currentEpoch++;
        imageLabelList = data.loadImages(images);
    }

    public boolean next(ImageBatch b) throws IOException {
        if (savedIterator == 0){
            newEpoch();
        }

        if (currentEpoch > totalEpochs){
            return false;
        }

        // clear the batch memory
        b.reset();

        for (int ii = savedIterator; ii < imageLabelList.size() ; ii++) {
            int i = ii % b.size;

            Pair<Integer, float[]> entry = imageLabelList.get(i);

            float[] image = entry.getSecond();
            Integer label = entry.getFirst();
            System.arraycopy(image, 0, b.images, i * image.length, image.length);
            b.labels[i * data.getNumClasses() + label] = (float) 1.0;

            i++;
            savedIterator++;
            if (i == b.size) {
                // exit when the batch is full
                return true;
            }
        }
        // we finished the list of data to add to the batch.
        savedIterator = 0;
        return false;
    }

    public boolean nextEpochs() {
        return currentEpoch <= totalEpochs;
    }
}
