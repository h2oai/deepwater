
#include "mlp.hpp"

using namespace std;
using namespace mxnet::cpp;

MLPNative::MLPNative() {
  ctx_dev = Context(DeviceType::kCPU, 0);
}

void MLPNative::setLayers(int * lsize, int nsize) {
  nLayers = nsize;
  for (int i = 0; i < nsize; i++) {
    layerSize.push_back(lsize[i]);
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
