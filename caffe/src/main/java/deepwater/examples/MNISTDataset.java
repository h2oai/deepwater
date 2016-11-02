package deepwater.examples;

import deepwater.Frame;
import deepwater.datasets.ImageDataSet;

public class MNISTDataset extends ImageDataSet {
    final Frame _frame;
    final Frame.Cursor _cursor;
    final int _batch;
    final float _scale = 0.00390625f;

    public MNISTDataset(String db, int batch) {
        super(28, 28, 3, 1000);
        _frame = new Frame.LMDB(db);
        _cursor = _frame.cursor();
        _batch = batch;
    }

    public void batch(float[] data, float[] labels) {
        for (int r = 0; r < _batch; r++) {
            Object[] cells = _cursor.cells();
            int row = getHeight() * getWidth();
            byte[] bytes = (byte[]) cells[0];
            if (bytes.length != row)
                throw new RuntimeException("Invalid dataset dimentions");
            for (int i = 0; i < bytes.length; ++i)
                data[r * row + i] = bytes[i] * _scale;
            labels[r] = (int) cells[1];
        }
    }
}



