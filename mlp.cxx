
#include "mlp.hpp"

using namespace std;
using namespace mxnet::cpp;

double score(mx_float* pred, mx_float* target, int dimY) {
  int right = 0;
  for (int i = 0; i < 128; ++i) {
    float mx_p = pred[i * 10 + 0];
    float p_y = 0;
    for (int j = 0; j < 10; ++j) {
      if (pred[i * 10 + j] > mx_p) {
        mx_p = pred[i * 10 + j];
        p_y = j;
      }
    }
    if (p_y == target[i]) right++;
  }
  cout << "Accuracy: " << right / dimY << endl;
  return 0.0;
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

void MLPClass::setAct(char ** acts) {
  for (int i = 0; i < nLayers; i++) {
    activations.push_back(acts[i]);
  }
}

void MLPClass::setX(float * aptr_x, int *, int) {
  array_x = NDArray(Shape(128, 28), ctx_dev, false);
  array_x.SyncCopyFromCPU(aptr_x, 128 * 28);
  array_x.WaitToRead();
}

void MLPClass::setLabel(float * aptr_y, int i) {
  array_y = NDArray(Shape(128), ctx_dev, false);
  array_y.SyncCopyFromCPU(aptr_y, 128);
  array_y.WaitToRead();
}

void MLPClass::buildnn() {

  for (int i = 0; i < nLayers; i++) {
    string istr = to_string(i);
    weights[i] = Symbol::Variable(string("w") + istr);
    biases[i] = Symbol::Variable(string("b") + istr);
    Symbol fc = FullyConnected(string("fc") + istr, 
                               i == 0? sym_x : outputs[i-1], 
                               weights[i], biases[i], layerSize[i]);
    outputs[i] = LeakyReLU(activations[i] + istr, fc, LeakyReLUActType::leaky);
  }
  sym_out = SoftmaxOutput("softmax", outputs[nLayers - 1], sym_label);
    // init the parameters
  NDArray array_w_1(Shape(512, 28), ctx_dev, false);
  NDArray array_b_1(Shape(512), ctx_dev, false);
  NDArray array_w_2(Shape(10, 512), ctx_dev, false);
  NDArray array_b_2(Shape(10), ctx_dev, false);

  // the parameters should be initialized in some kind of distribution,
  // so it learns fast
  // but here just give a const value by hand
  array_w_1 = 0.5f;
  array_b_1 = 0.0f;
  array_w_2 = 0.5f;
  array_b_2 = 0.0f;

  // the grads
  NDArray array_w_1_g(Shape(512, 28), ctx_dev, false);
  NDArray array_b_1_g(Shape(512), ctx_dev, false);
  NDArray array_w_2_g(Shape(10, 512), ctx_dev, false);
  NDArray array_b_2_g(Shape(10), ctx_dev, false);

  // Bind the symolic network with the ndarray
  // all the input args
  
  in_args.push_back(array_x);
  in_args.push_back(array_w_1);
  in_args.push_back(array_b_1);
  in_args.push_back(array_w_2);
  in_args.push_back(array_b_2);
  in_args.push_back(array_y);
  // all the grads
  
  arg_grad_store.push_back(NDArray());  // we don't need the grad of the input
  arg_grad_store.push_back(array_w_1_g);
  arg_grad_store.push_back(array_b_1_g);
  arg_grad_store.push_back(array_w_2_g);
  arg_grad_store.push_back(array_b_2_g);
  arg_grad_store.push_back(NDArray());  // neither do we need the grad of the loss
  // how to handle the grad
  
  grad_req_type.push_back(kNullOp);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kNullOp);
  

  exe = std::make_shared<Executor>(sym_out, ctx_dev, in_args, arg_grad_store,
                                   grad_req_type, aux_states);
}

float MLPClass::train(bool verbose) {

  exe->Forward(true);

  if (verbose) {
    std::vector<NDArray>& out = exe->outputs;
    float* cptr = new float[128 * 10];
    out[0].SyncCopyToCPU(cptr, 128 * 10);
    NDArray::WaitAll();

    float* aptr_y = new float[128];
    array_y.SyncCopyToCPU(aptr_y, 128);
    NDArray::WaitAll();
    delete[] cptr;
    delete[] aptr_y;
    return score(cptr, aptr_y, 128);
  }

  // update the parameters
  exe->Backward();
  for (int i = 1; i < 5; ++i) {
    in_args[i] -= arg_grad_store[i] * learning_rate;
  }
  NDArray::WaitAll();
}
