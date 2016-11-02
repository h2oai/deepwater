package deepwater.backends.caffe;

import org.bytedeco.javacpp.caffe;

import deepwater.backends.BackendModel;
import deepwater.backends.BackendParams;
import deepwater.backends.BackendTrain;
import deepwater.backends.RuntimeOptions;
import deepwater.datasets.ImageDataSet;

public class CaffeBackend implements BackendTrain {
    @Override
    public void delete(BackendModel m) {
        ((Model) m).close();
    }

    @Override
    public BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts, BackendParams backend_params, int num_classes, String name) {
        caffe.SolverParameter solver = new caffe.SolverParameter();
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
        solver.set_snapshot(0);
        solver.set_snapshot_prefix("snapshots");
        solver.set_solver_mode(caffe.Caffe.GPU);

        caffe.NetParameter net = caffe.NetParameter.default_instance().New();
        caffe.ReadProtoFromTextFileOrDie((String) backend_params.get("graph"), net);
        solver.set_allocated_net_param(net);
        return new Model(solver);
    }

    @Override
    public void saveModel(BackendModel m, String model_path) {
        // Why do we need that?
    }

    @Override
    public void loadParam(BackendModel m, String param_path) {
        ((Model) m).load(param_path);
    }

    @Override
    public void saveParam(BackendModel m, String param_path) {
        ((Model) m).save(param_path);
    }

    @Override
    public float[] loadMeanImage(BackendModel m, String path) {
        return new float[0];
    }

    @Override
    public String toJson(BackendModel m) {
        return null;
    }

    @Override
    public void setParameter(BackendModel m, String name, float value) {
        if ("momentum".equals(name)) {
            ((Model) m).setMomentum(value);
        } else if ("learning_rate".equals(name)) {
            ((Model) m).setLearningRate(value);
        } else if ("clip_gradient".equals(name)) {
//            ((Model) m).setClipGradient(value);
        } else throw new IllegalArgumentException("invalid parameter: " + name);
    }

    @Override
    public float[] train(BackendModel m, float[] data, float[] label) {
        ((Model) m).feed(data, label);
        return null;
    }

    @Override
    public float[] predict(BackendModel m, float[] data, float[] label) {
        return new float[0];
    }

    @Override
    public float[] predict(BackendModel m, float[] data) {
        return ((Model) m).predict(data);
    }
}
