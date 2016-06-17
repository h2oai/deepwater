
#include "mlp.hpp"

using namespace std;
using namespace mxnet::cpp;

float score(float* pred, std::vector<float> target, int dimY, int nout) {
  int right = 0;
  for (int i = 0; i < dimY; ++i) {
    float mx_p = pred[i * nout + 0];
    float p_y = 0;
    for (int j = 0; j < nout; ++j) {
      if (pred[i * nout + j] > mx_p) {
        mx_p = pred[i * nout + j];
        p_y = j;
      }
    }
    if (p_y == target[i]) right++;
  }
  return static_cast<float>(right) / dimY;
}

MLPClass::MLPClass() {
  sym_x = Symbol::Variable("X");
  sym_label = Symbol::Variable("label");
  ctx_dev = Context(DeviceType::kCPU, 0);
  dimX1 = 0;
  dimX2 = 0;
  dimY = 0;
}

void MLPClass::setLayers(int * lsize, int nsize) {
  nLayers = nsize;
  for (int i = 0; i < nsize; i++) {
    layerSize.push_back(lsize[i]);
  }
  weights.resize(nLayers);
  biases.resize(nLayers);
  outputs.resize(nLayers);
}

void MLPClass::setX(float * aptr_x, int * dims, int n_dim) {
  if (n_dim == 1) {
    dimX1 = dims[0];
    dimX2 = 1;
    array_x = NDArray(Shape(dimX1), ctx_dev, false);
  } else if (n_dim == 2) {
    dimX1 = dims[0];
    dimX2 = dims[1];
    array_x = NDArray(Shape(dimX1, dimX2), ctx_dev, false);
  }
  array_x.SyncCopyFromCPU(aptr_x, dimX1 * dimX2);
  array_x.WaitToRead();
}

void MLPClass::setLabel(float * aptr_y, int i) {
  dimY = i;
  array_y = NDArray(Shape(dimY), ctx_dev, false);
  array_y.SyncCopyFromCPU(aptr_y, dimY);
  array_y.WaitToRead();
  label.assign(aptr_y, aptr_y + dimY);
}

void MLPClass::buildnn() {

  for (int i = 0; i < nLayers; i++) {
    string istr = to_string(i);
    weights[i] = Symbol::Variable(string("w") + istr);
    biases[i] = Symbol::Variable(string("b") + istr);
    Symbol fc = FullyConnected(string("fc") + istr, 
                               i == 0? sym_x : outputs[i-1], 
                               weights[i], biases[i], layerSize[i]);
    outputs[i] = LeakyReLU(string("act") + istr, fc, LeakyReLUActType::leaky);
  }
  sym_out = SoftmaxOutput("softmax", outputs[nLayers - 1], sym_label);

  // init the parameters
  in_args.push_back(array_x);
  for (int i = 0; i < nLayers;i++) {
    NDArray array_w;
    if (i == 0) {
      array_w = NDArray(Shape(layerSize[i], dimX2), ctx_dev, false);
    } else {
      array_w = NDArray(Shape(layerSize[i], layerSize[i - 1]), ctx_dev, false);
    }
    NDArray array_b(Shape(layerSize[i]), ctx_dev, false);
    // maybe they should be set according to some distributions
    array_w = 0.5f;
    array_b = 0.0f;
    in_args.push_back(array_w);
    in_args.push_back(array_b);
  }
  in_args.push_back(array_y);

  // the grads
  arg_grad_store.push_back(NDArray());
  for (int i = 0; i < nLayers; i++) {
    NDArray array_w_g;
    if (i == 0) {
      array_w_g = NDArray(Shape(layerSize[i], dimX2), ctx_dev, false); 
    } else {
      array_w_g = NDArray(Shape(layerSize[i], layerSize[i - 1]), ctx_dev, false); 
    }
    arg_grad_store.push_back(array_w_g);
    NDArray array_b_g(Shape(layerSize[i]), ctx_dev, false);
    arg_grad_store.push_back(array_b_g);
  }
  arg_grad_store.push_back(NDArray());

  // handle the grad
  grad_req_type.push_back(kNullOp);
  for (int i = 0; i < nLayers; i++) {
    grad_req_type.push_back(kWriteTo);
    grad_req_type.push_back(kWriteTo);
  }
  grad_req_type.push_back(kNullOp);
  exe = std::make_shared<Executor>(sym_out, ctx_dev, in_args, arg_grad_store,
                                   grad_req_type, aux_states);
}

float MLPClass::train(int iter, bool verbose) {

  float acc = 0.0;
  pred = (float *)malloc(sizeof(float) * dimY * layerSize[layerSize.size() - 1]);

  for (int i = 0; i < iter; i++) {
    exe->Forward(true);
    exe->Backward();
    for (int i = 1; i < 5; ++i) {
      in_args[i] -= arg_grad_store[i] * learning_rate;
    }
    NDArray::WaitAll();
  }

  if (verbose) {
    std::vector<NDArray>& out = exe->outputs;
    out[0].SyncCopyToCPU(pred, dimY * layerSize[layerSize.size() - 1]);
    NDArray::WaitAll();
    acc = score(pred, label, dimY, layerSize[layerSize.size() - 1]);
  }

  free(pred);
  return acc;
}
