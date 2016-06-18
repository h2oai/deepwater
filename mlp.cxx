
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
}

void MLPNative::setAct(char ** acts) {
  for (int i = 0; i < nLayers; i++) {
    activations.push_back(acts[i]);
  }
}

void MLPNative::setData(float * aptr_x, int * dims, int n_dim) {
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
  //for (auto s : sym_network.ListArguments()) {
  //  std::cout << s << std::endl;  
  //}
  //std::cout <<
  sym_network.Save("mlp.json");
}

mx_float* MLPNative::train() {

  pred = (float *)malloc(sizeof(float) * dimY * layerSize[layerSize.size() - 1]);

  exe->Forward(true);
  exe->Backward();
  Optimizer opt("ccsgd", learning_rate, weight_decay);
  exe->UpdateAll(&opt, learning_rate, weight_decay);
  NDArray::WaitAll();

  std::vector<NDArray>& out = exe->outputs;
  out[0].SyncCopyToCPU(pred, dimY * layerSize[layerSize.size() - 1]);
  NDArray::WaitAll();

  return pred;
}
