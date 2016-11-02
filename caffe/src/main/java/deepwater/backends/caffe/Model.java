package deepwater.backends.caffe;

import deepwater.backends.BackendModel;
import org.bytedeco.javacpp.caffe;
import org.bytedeco.javacpp.caffe.*;

import java.io.Closeable;

public class Model implements BackendModel, Closeable {
    private final FloatSolver _solver;
    private final FloatNet _net;
    private final Object _lock = new Object();

    public Model(SolverParameter params) {
        Caffe.set_mode(Caffe.GPU);
        Caffe.SetDevice(params.device_id());
        _solver = FloatSolverRegistry.CreateSolver(params);
        _net = _solver.net();
    }

    @Override
    public void close() {
        _solver.close();
    }

    public void setLearningRate(float value) {
        _solver.param().set_base_lr(value);
    }

    public void setMomentum(float value) {
        _solver.param().set_momentum(value);
    }

    public void save(String file) {
        _net.ToHDF5(file);
    }

    public void load(String file) {
        _net.CopyTrainedLayersFromHDF5(file);
    }

    public void feed(float[] data, float[] label) {
        synchronized (_lock) {
            FloatBlob d = _net.input_blobs().get(0);
            d.cpu_data().put(data);

            FloatBlob l = _net.input_blobs().get(1);
            l.cpu_data().put(label);

            _solver.Step(1);
        }
    }

    public float[] predict(float[] data) {
        synchronized (_lock) {
            caffe.FloatBlobVector blobs = _net.Forward(data);
            float[] out = new float[blobs.get(0).count()];
            blobs.get(0).cpu_data().get(out);
            return out;
        }
    }
}