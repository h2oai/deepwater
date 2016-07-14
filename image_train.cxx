/*!
 * Copyright (c) 2016 by Contributors
 */
#include <vector>
#include <string>
#include <cassert>

#include "include/symbol.h"
#include "include/optimizer.h"
#include "image_train.hpp"

using namespace mxnet::cpp;

ImageTrain::ImageTrain() {
  width = 224;
  height = 224;
  learning_rate = 0.001;
  weight_decay = 1e-4;
  is_built = false;
}

ImageTrain::~ImageTrain() {
  // delete exec;
}

void ImageTrain::buildNet(int n, int b, char * n_name) {
  std::string net_name(n_name);
  assert(net_name == "inception_bn" ||
         net_name == "vgg" ||
         net_name == "lenet" ||
         net_name == "alexnet" ||
         net_name == "googlenet" ||
         net_name == "resnet");

  if (net_name == "inception_bn") {
    mxnet_sym = InceptionSymbol(n);
  } else if (net_name == "vgg") {
    mxnet_sym = VGGSymbol(n);
  } else if (net_name == "lenet") {
    mxnet_sym = LenetSymbol(n);
  } else if (net_name == "alexnet") {
    mxnet_sym = AlexnetSymbol(n);
  } else if (net_name == "googlenet") {
    mxnet_sym = GoogleNetSymbol(n);
  } else {
    mxnet_sym = ResNetSymbol(n);
  }

  batch_size = b;
  num_classes = n;

  opt = new Optimizer("ccsgd", learning_rate, weight_decay);
  opt->SetParam("momentum", 0.9);
  opt->SetParam("rescale_grad", 1.0 / batch_size);
  opt->SetParam("clip_gradient", 10);

  args_map["data"] = NDArray(Shape(batch_size, 3, width, height), Context::gpu());
  args_map["data_label"] = NDArray(Shape(batch_size), Context::gpu());
  mxnet_sym.InferArgsMap(Context::gpu(), &args_map, args_map);
  exec = mxnet_sym.SimpleBind(Context::gpu(), args_map);
  is_built = true;
}

void ImageTrain::loadModel(char * model_path) {
  mxnet_sym = Symbol::LoadJSON(std::string(model_path));
  is_built = true;
}

void ImageTrain::loadParam(char * param_path) {
  NDArray::Load(std::string(param_path), nullptr, &args_map);
}

void ImageTrain::saveParam(char * param_path) {
  NDArray::Save(std::string(param_path), args_map);
}

std::vector<float> ImageTrain::train(float * data, float * label) {
  return execute(data, label, true);
}

std::vector<float> ImageTrain::predict(float * data, float * label) {
  return execute(data, label, false);
}


std::vector<float> ImageTrain::execute(float * data, float * label, bool is_train) {
  if (!is_built) {
    std::cerr << "Network hasn't been built. "
        << "Please run buildNet() or loadModel() first." << std::endl;
    exit(0);
  }
  NDArray data_n = NDArray(data, Shape(batch_size, 3, width, height), Context::gpu());

  NDArray label_n = NDArray(label, Shape(batch_size), Context::gpu());

  data_n.CopyTo(&args_map["data"]);
  label_n.CopyTo(&args_map["data_label"]);

  NDArray::WaitAll();

  exec->Forward(is_train);
  // train or predict?
  if (is_train) {
    exec->Backward();
    exec->UpdateAll(opt, learning_rate, weight_decay);
  }

  NDArray::WaitAll();

  // get probs for prediction
  std::vector<float> preds(batch_size * num_classes);
  exec->outputs[0].SyncCopyToCPU(&preds, batch_size * num_classes);

  return preds;
}
