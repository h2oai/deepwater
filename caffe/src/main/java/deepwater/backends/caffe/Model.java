package deepwater.backends.caffe;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.americano;
import org.bytedeco.javacpp.americano.Barrier;
import org.bytedeco.javacpp.americano.FloatNCCL;
import org.bytedeco.javacpp.americano.NCCLVector;
import org.bytedeco.javacpp.caffe;
import org.bytedeco.javacpp.caffe.Caffe;
import org.bytedeco.javacpp.caffe.FloatBlob;
import org.bytedeco.javacpp.caffe.FloatNet;
import org.bytedeco.javacpp.caffe.FloatSolver;
import org.bytedeco.javacpp.caffe.FloatSolverRegistry;
import org.bytedeco.javacpp.caffe.SolverParameter;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;

import deepwater.backends.BackendModel;

public class Model implements BackendModel, Closeable {
    private final int[] _gpus;
    private final Barrier _barrier, _barrier_init;
    private final ArrayList<SolverThread> _solvers;
    private final Object _lock = new Object();
    private int _rank;
    private volatile float _lr;
    private volatile float _momentum;

    public Model(SolverParameter solver) {
        _gpus = new int[americano.device_count()];
        for (int i = 0; i < americano.device_count(); i++)
            _gpus[i] = i;

        // Barriers for NCCL init and runtime
        _barrier_init = new Barrier(1 + _gpus.length);
        _barrier = new Barrier(_gpus.length);

        // Create a solver per GPU
        _solvers = new ArrayList<>();
        for (int rank = 0; rank < _gpus.length; rank++) {
            SolverThread st = new SolverThread(solver, rank);
            // TODO core affinity
            st.start();
            _solvers.add(st);
        }

        // Wait for solvers creation
        _barrier_init.Wait();

        // Init NCCL
        NCCLVector nccls = new NCCLVector();
        nccls.resize(_gpus.length);
        for (int i = 0; i < _gpus.length; i++)
            nccls.put(i, _solvers.get(i)._nccl);
        FloatNCCL.InitSingleProcess(nccls);
        _barrier_init.Wait();
    }

    @Override
    public void close() {
        for (SolverThread thread : _solvers) {
            thread._stop = true;
            thread.interrupt();
            try {
                thread.join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void setLearningRate(float value) {
        _lr = value;
    }
    public void setMomentum(float value) {
        _momentum = value;
    }
    public void save(String file) {
        _solvers.get(0)._solver.net().ToHDF5(file);
    }
    public void load(String file) {
        _solvers.get(0)._solver.net().CopyTrainedLayersFromHDF5(file);
    }

    public void feed(float[] data, float[] label) {
        int rank;
        synchronized (_lock) {
            rank = _rank;
            _rank++;
            if (_rank == _solvers.size())
                _rank = 0;
        }
        try {
            Map<String, FloatBlob> map = _solvers.get(rank)._free.take();
            // Hardcoded names for now
            caffe.FloatBlob data_blob = map.get("data");
            caffe.FloatBlob labs_blob = map.get("label");
            data_blob.cpu_data().put(data);
            labs_blob.cpu_data().put(label);
            // Prefetch to GPU memory
            americano.set_device(_gpus[rank]);
            data_blob.gpu_data();
            labs_blob.gpu_data();
            _solvers.get(rank)._full.put(map);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public float[] predict(float[] data) {
        caffe.FloatBlobVector blobs = _solvers.get(0)._solver.net().Forward(data);
        float[] out = new float[blobs.get(0).count()];
        blobs.get(0).cpu_data().get(out);
        return out;
    }

    class SolverThread extends Thread {
        final SolverParameter _proto;
        final int _rank;
        FloatNCCL _nccl;
        volatile FloatSolver _solver;
        volatile boolean _stop;

        static final int PREFETCH = 4;
        final ArrayBlockingQueue<Map<String, FloatBlob>> _free;
        final ArrayBlockingQueue<Map<String, FloatBlob>> _full;

        SolverThread(SolverParameter proto, int rank) {
            _proto = new SolverParameter(proto);
            _proto.set_device_id(_gpus[rank]);
            _rank = rank;
            _free = new ArrayBlockingQueue<>(PREFETCH);
            _full = new ArrayBlockingQueue<>(PREFETCH);
        }

        @Override
        public void run() {
            try {
                americano.set_device(_gpus[_rank]);
                Caffe.set_mode(Caffe.GPU);
                Caffe.set_solver_count(_gpus.length);
                Caffe.set_solver_rank(_rank);

                _solver = FloatSolverRegistry.CreateSolver(_proto);
                FloatNet net = _solver.net();

                String[] inputs = new String[net.num_inputs()];
                for (int i = 0; i < net.num_inputs(); i++) {
                    int blob_index = net.input_blob_indices().get(i);
                    inputs[i] = net.blob_names().get(blob_index).getString();
                }

                for (int p = 0; p < PREFETCH; p++) {
                    Map<String, FloatBlob> map = new HashMap<>();
                    for (int i = 0; i < net.num_inputs(); i++) {
                        FloatBlob blob = new FloatBlob();
                        blob.Reshape(net.input_blobs().get(i).shape());
                        map.put(inputs[i], blob);
                    }
                    _free.add(map);
                }

                _nccl = new FloatNCCL(_solver, _barrier);
                // Wait for other threads
                _barrier_init.Wait();
                // Wait for shared NCCL init
                _barrier_init.Wait();
                _nccl.Broadcast();

                for (int it = 0; !_stop && it < _proto.max_iter(); it++) {
                    Map<String, FloatBlob> map = _full.poll();
                    if (map == null) {
//                        System.out.println("Waiting on data");
                        map = _full.take();
                    }
                    for (int i = 0; i < inputs.length; i++) {
                        FloatBlob a = map.get(inputs[i]);
                        FloatBlob b = net.input_blobs().get(i);
                        b.set_gpu_data(a.gpu_data());
                    }
                    _solver.param().set_lr(_learning_rate);
                    _solver.param().set_momentum(_momentum);
                    _solver.Step(1);
                    _free.put(map);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
