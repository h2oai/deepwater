package deepwater.examples;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;

import deepwater.Frame;
import deepwater.datasets.ImageDataSet;

public class ImageNetDataset extends ImageDataSet {
    final Frame _frame;
    final Frame.Cursor _cursor;
    final opencv_core.MatVector _mats;
    final float[] _mean = {104, 117, 123};

    public ImageNetDataset(String db, int batch) {
        super(224, 224, 3, 1000);
        _frame = new Frame.LMDB(db);
        _cursor = _frame.cursor();
        _mats = new opencv_core.MatVector(batch);
    }

    public void batch(float[] data, float[] labels) {
        final int batch = (int) _mats.size();
        for (int i = 0; i < batch; i++) {
            Object[] cells = _cursor.cells();
            opencv_core.Mat encoded = new opencv_core.Mat(new BytePointer((byte[]) cells[0]));
            opencv_core.Mat decoded = opencv_imgcodecs.imdecode(encoded,
                opencv_imgcodecs.CV_LOAD_IMAGE_COLOR);
            if (decoded.arrayWidth() != 0) {
                opencv_imgproc.resize(decoded, decoded,
                    new opencv_core.Size(getWidth(), getHeight()));
                _mats.put(i, decoded);
                labels[i] = (int) cells[1];
            }
            _cursor.next();
        }

        int matSize = _mats.get(0).arraySize();
        for (int i = 0; i < batch; i++) {
            for (int h = 0; h < getHeight(); ++h) {
                BytePointer ptr = _mats.get(i).ptr(h);
                int img_index = 0;
                for (int w = 0; w < getWidth(); ++w) {
                    for (int c = 0; c < getChannels(); ++c) {
                        int top_index = (c * getHeight() + h) * getWidth() + w;
                        float pixel = (float) ptr.get(img_index++);
                        data[i * matSize + top_index] = (pixel - _mean[c]);
                    }
                }
            }
        }
    }
}



