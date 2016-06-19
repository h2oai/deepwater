
#include "mlp.hpp"

using namespace std;
using namespace mxnet::cpp;

MLPNative::MLPNative() {
  ctx_dev = Context(DeviceType::kCPU, 0);
}

void MLPNative::setLayers(int * lsize, int nsize, int n) {
  nLayers = nsize;
  nOut = n;
  for (int i = 0; i < nsize; i++) {
    layerSize.push_back(lsize[i]);
  }
  pred = (float *)malloc(sizeof(float) * dimY * nOut);
}

void MLPNative::setAct(char ** acts) {
  for (int i = 0; i < nLayers; i++) {
    activations.push_back(acts[i]);
  }
}

void MLPNative::setData(float * aptr_x, int dim1, int dim2) {
  dimX1 = dim1;
  dimX2 = dim2;
  array_x = NDArray(Shape(dimX1, dimX2), ctx_dev, false);
  array_x.SyncCopyFromCPU(aptr_x, dimX1 * dimX2);
  array_x.WaitToRead();
}

void MLPNative::setLabel(float * aptr_y, int i) {
  dimY = i;
  array_y = NDArray(Shape(dimY), ctx_dev, false);
  array_y.SyncCopyFromCPU(aptr_y, dimY);
  array_y.WaitToRead();
  label.assign(aptr_y, aptr_y + dimY);
}

void MLPNative::build_mlp() {
  Symbol act = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");
  std::vector<Symbol> fc_w, fc_b, fc;

  for (int i = 0; i < nLayers; i++) {
    fc_w.push_back(Symbol("fc" + std::to_string(i + 1) + "_w"));
    fc_b.push_back(Symbol("fc" + std::to_string(i + 1) + "_b"));
    fc.push_back(FullyConnected("fc" + std::to_string(i + 1), 
                                act, fc_w[i], fc_b[i], layerSize[i]));
    act = Activation("act" + std::to_string(i + 1),
                     fc[i], activations[i]);
  }
  fc_w.push_back(Symbol("fc" + std::to_string(nLayers + 1) + "_w"));
  fc_b.push_back(Symbol("fc" + std::to_string(nLayers + 1) + "_b"));
  fc.push_back(FullyConnected("fc" + std::to_string(nLayers + 1),
                              act, fc_w[nLayers], fc_b[nLayers], nOut));
  sym_network = SoftmaxOutput("softmax", fc[nLayers], data_label);
  sym_network.Save("mlp.json");
  args_map["data"] = array_x.Slice(0, batch_size).Copy(ctx_dev);
  args_map["data_label"] = array_y.Slice(0, batch_size).Copy(ctx_dev);
  sym_network.InferArgsMap(ctx_dev, &args_map, args_map);
  //opt = Optimizer("ccsgd", learning_rate, weight_decay);
  //opt.SetParam("momentum", 0.9)
  //    .SetParam("rescale_grad", 1.0)
  //    .SetParam("clip_gradient", 10);
}

mx_float* MLPNative::train() {

  Optimizer opt("ccsgd", learning_rate, weight_decay);
  opt.SetParam("momentum", 0.9)
      .SetParam("rescale_grad", 1.0)
      .SetParam("clip_gradient", 10);

  int start_index = 0;
  while (start_index < dimY) {
    if (start_index + batch_size > dimY) {
      start_index = dimY - batch_size; 
    } 
    args_map["data"] = array_x.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
    args_map["data_label"] = array_y.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
    start_index += batch_size;
    NDArray::WaitAll();

    Executor * exe = sym_network.SimpleBind(ctx_dev, args_map);
    exe->Forward(true);
    exe->Backward();
    exe->UpdateAll(&opt, learning_rate, weight_decay);
    NDArray::WaitAll();
    if (start_index == dimY - batch_size) {
      std::vector<NDArray> & out = exe->outputs; 
      out[0].SyncCopyToCPU(pred, dimY * layerSize[layerSize.size() - 1]);
      NDArray::WaitAll();
    }
  }

  return pred;
}
