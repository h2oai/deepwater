/*!
 * Copyright (c) 2016 by Contributors
 */
#include <vector>
#include <string>
#include <cassert>
#include "mlp.hpp"

using namespace std;
using namespace mxnet::cpp;

MLP::MLP() {
  batch_size = -1;
#if MSHADOW_USE_CUDA == 0
  ctx_dev = Context(DeviceType::kCPU, 0);
#else
  ctx_dev = Context(DeviceType::kGPU, 0);
#endif
}

MLP::~MLP() {
  delete exec;
  delete opt;
}

void MLP::setLayers(int * lsize, int nsize, int n) {
  nLayers = nsize;
  num_classes = n;
  for (int i = 0; i < nsize; i++) {
    layerSize.push_back(lsize[i]);
  }
}

void MLP::setAct(char ** acts) {
  for (int i = 0; i < nLayers; i++) {
    activations.push_back(acts[i]);
  }
}

void MLP::saveParam(char * param_path) {
  NDArray::Save(std::string(param_path), args_map);
}

void MLP::loadParam(char * param_path) {
  NDArray::Load(std::string(param_path), nullptr, &args_map);
}

void MLP::saveModel(char * model_path) {
  if (is_built)
    sym_network.Save(std::string(model_path));
  else
    std::cerr << "Network hasn't been built" << std::endl;
}

void MLP::loadModel(char * model_path) {
  sym_network = Symbol::Load(std::string(model_path));
  is_built = true;
}

void MLP::buildMLP() {
  if (batch_size == -1) std::cerr << "Batch size not set yet" << std::endl;
  Symbol act = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");
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
                              act, fc_w[nLayers], fc_b[nLayers], num_classes));

  sym_network = SoftmaxOutput("softmax", fc[nLayers], data_label);
  sym_network.InferArgsMap(ctx_dev, &args_map, args_map);
  exec = sym_network.SimpleBind(ctx_dev, args_map);
  is_built = true;
  preds.resize(num_classes * batch_size);
}

void MLP::setSeed(int seed) {
  MXRandomSeed(seed);
}

std::vector<float> MLP::train(float * data, float * label) {
  return execute(data, label, true);
}

std::vector<float> MLP::predict(float * data, float * label) {
  return execute(data, label, false);
}

std::vector<float> MLP::predict(float * data) {
  return execute(data, NULL, false);
}

std::vector<float> MLP::execute(float * data, float * label, bool is_train) {
  if (!is_built) {
    std::cerr << "Network hasn't been built. "
        << "Please run buildNet() or loadModel() first." << std::endl;
    exit(0);
  }

  NDArray data_n = NDArray(data, Shape(batch_size, 1, dimX), ctx_dev);
  data_n.CopyTo(&args_map["data"]);

  if (is_train) {
    NDArray label_n = NDArray(label, Shape(batch_size), ctx_dev);
    label_n.CopyTo(&args_map["softmax_label"]);
  }
  NDArray::WaitAll();

  exec->Forward(is_train);
  // train or predict?
  if (is_train) {
    exec->Backward();
    exec->UpdateAll(opt, learning_rate, weight_decay);
  }

  NDArray::WaitAll();

  // get probs for prediction
  exec->outputs[0].SyncCopyToCPU(&preds, batch_size * num_classes);
  return preds;
}
