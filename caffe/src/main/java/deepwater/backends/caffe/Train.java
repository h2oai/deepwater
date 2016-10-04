package deepwater.backends.caffe;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.americano;
import org.bytedeco.javacpp.americano.Barrier;
import org.bytedeco.javacpp.americano.FloatNCCL;
import org.bytedeco.javacpp.americano.NCCLVector;
import org.bytedeco.javacpp.caffe;
import org.bytedeco.javacpp.caffe.Caffe;
import org.bytedeco.javacpp.caffe.FloatBlob;
import org.bytedeco.javacpp.caffe.FloatDataTransformer;
import org.bytedeco.javacpp.caffe.FloatNet;
import org.bytedeco.javacpp.caffe.FloatSolver;
import org.bytedeco.javacpp.caffe.FloatSolverRegistry;
import org.bytedeco.javacpp.caffe.InputParameter;
import org.bytedeco.javacpp.caffe.LayerParameter;
import org.bytedeco.javacpp.caffe.NetParameter;
import org.bytedeco.javacpp.caffe.SolverParameter;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;

import deepwater.backends.caffe.Frame.Cursor;

public class Train {
  public static void main(String[] args) throws Exception {
//    System.setProperty("org.bytedeco.javacpp.logger.debug", "true");

//    H2O.main(new String[]{"-name", "americano"});
//    H2O.waitForCloudSize(1, 0);
//    Frame frame = water.parser.ParseDataset.parse(Key.make(), nfs._key);
    final Frame frame = new Frame.LMDB("/datasets/imagenet_1000_train/");

    SolverParameter solver = new SolverParameter();
    solver.add_test_iter(1000);
    solver.set_test_interval(Integer.MAX_VALUE);  // Disable tests
    solver.set_test_initialization(false);
    solver.set_display(40);
    solver.set_average_loss(40);
    solver.set_base_lr(0.01f);
    solver.set_lr_policy("step");
    solver.set_stepsize(320000);
    solver.set_gamma(0.96f);
    solver.set_max_iter(10000000);
    solver.set_momentum(0.9f);
    solver.set_weight_decay(0.0002f);
    solver.set_layer_wise_reduce(true);
    solver.set_snapshot(0);
    solver.set_snapshot_prefix("snapshots");
    solver.set_solver_mode(Caffe.GPU);

    NetParameter net = new NetParameter();
    caffe.ReadProtoFromTextFileOrDie(
        "examples/googlenet.prototxt", net);
    solver.set_allocated_net_param(net);
    LayerParameter layer = net.mutable_layer(0);
    InputParameter input = layer.input_param();
    final int batch = (int) input.shape(0).dim(0);

    int[] gpus = new int[americano.device_count()];

    // Create a solver per GPU
    for (int i = 0; i < americano.device_count(); i++) {
         gpus[i] = i;
    }
    Barrier barrier_init = new Barrier(1 + gpus.length);
    Barrier barrier = new Barrier(gpus.length);
    final ArrayList<SolverThread> solvers = new ArrayList<>();
    final int cpus = Runtime.getRuntime().availableProcessors();
    int cpu = 0;
    for (int i = 0; i < gpus.length; i++) {
      SolverThread st = new SolverThread(
          solver, gpus, i, barrier_init, barrier);
      // TODO core affinity
      st.start();
      solvers.add(st);

      // Create data threads, that could be done in H2O
      for (int j = 0; j < cpus / gpus.length; j++) {
        final int gpu_rank = i;
        final int cpu_rank = cpu++;
        final FloatDataTransformer transformer =
            new FloatDataTransformer(layer.transform_param(), caffe.TRAIN);
        transformer.InitRand();
        Thread thread = new Thread() {
          @Override
          public void run() {
            americano.set_device(gpus[gpu_rank]);
            Caffe.set_mode(Caffe.GPU);
            MatVector mats = new MatVector(batch);
            Cursor cursor = frame.cursor();
            long row = 0;
            try {
              //noinspection InfiniteLoopStatement
              for (; ; ) {
                Map<String, FloatBlob> map = solvers.get(gpu_rank)._free.take();
                // Hardcoded names for now
                FloatBlob data = map.get("data");
                FloatBlob labs = map.get("label");
                FloatPointer labs_cpu = labs.cpu_data();

                for (int count = 0; count < batch; count++) {
                  // One row per CPU - deterministic load balance
                  while ((row++ % cpus) != cpu_rank)
                    cursor.next();
                  Object[] cells = cursor.cells();
                  Mat encoded = new Mat(new BytePointer((byte[]) cells[0]));
                  Mat decoded = opencv_imgcodecs.imdecode(encoded,
                      opencv_imgcodecs.CV_LOAD_IMAGE_COLOR);
                  if (decoded.arrayWidth() != 0) {
                    opencv_imgproc.resize(decoded, decoded, new opencv_core.Size(
                        256, 256));
                    mats.put(count, decoded);
                    labs_cpu.put(count, (int) cells[1]);
                  }
                  cursor.next();
                }
                transformer.Transform(mats, data);
                // Prefetch to GPU memory
                data.gpu_data();
                labs.gpu_data();
                solvers.get(gpu_rank)._full.put(map);
              }
            } catch (InterruptedException ex) {
              // Ignore
            }
          }
        };
        thread.setDaemon(true);
        thread.start();
      }
    }

    // Wait for solvers creation
    barrier_init.Wait();

    // Init NCCL
    NCCLVector nccls = new NCCLVector();
    nccls.resize(gpus.length);
    for (int i = 0; i < gpus.length; i++)
      nccls.put(i, solvers.get(i)._nccl);
    FloatNCCL.InitSingleProcess(nccls);
    barrier_init.Wait();

    for (Thread thread : solvers)
      thread.join();
  }

  static class SolverThread extends Thread {
    final SolverParameter _proto;
    final int[] _gpus;
    final int _rank;
    final Barrier _barrier_init, _barrier;
    FloatNCCL _nccl;

    static final int PREFETCH = 4;
    final ArrayBlockingQueue<Map<String, FloatBlob>> _free;
    final ArrayBlockingQueue<Map<String, FloatBlob>> _full;

    SolverThread(SolverParameter proto, int[] gpus, int s,
                 Barrier barrier_init, Barrier barrier) {
      _proto = new SolverParameter(proto);
      _proto.set_device_id(gpus[s]);
      _gpus = gpus;
      _rank = s;
      _barrier_init = barrier_init;
      _barrier = barrier;
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

        FloatSolver solver = FloatSolverRegistry.CreateSolver(_proto);
        FloatNet net = solver.net();

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
            int[] shape = new int[(int) net.input_blobs().get(i).shape().limit()];
            net.input_blobs().get(i).shape().get(shape);
            map.put(inputs[i], blob);
          }
          _free.add(map);
        }

        _nccl = new FloatNCCL(solver, _barrier);
        // Wait for other threads
        _barrier_init.Wait();
        // Wait for shared NCCL init
        _barrier_init.Wait();
        _nccl.Broadcast();

        for (int it = 0; it < _proto.max_iter(); it++) {
          Map<String, FloatBlob> map = _full.poll();
          if (map == null) {
            System.out.println("Waiting on data");
            map = _full.take();
          }
          for (int i = 0; i < inputs.length; i++) {
            FloatBlob a = map.get(inputs[i]);
            FloatBlob b = net.input_blobs().get(i);
            b.set_gpu_data(a.gpu_data());
          }
          solver.Step(1);
          _free.put(map);
        }
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }
  }
}
