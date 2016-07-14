/*!
 * Copyright (c) 2016 by Contributors
 */
#include <vector>
#include <string>
#include "include/optimizer.h"
#include "image_classify.hpp"

using namespace mxnet::cpp;

ImageClassify::ImageClassify() {
  width = 224;
  height = 224;
  learning_rate = 0.001;
  weight_decay = 1e-4;
}

ImageClassify::~ImageClassify() {
  // delete exec;
}

void ImageClassify::buildNet(int n, int b) {
  inception_bn_net = InceptionSymbol(n);
  batch_size = b;
  num_classes = n;

  opt = new Optimizer("ccsgd", learning_rate, weight_decay);
  (*opt).SetParam("momentum", 0.9);
  (*opt).SetParam("rescale_grad", 1.0 / batch_size);
  (*opt).SetParam("clip_gradient", 10);

  args_map["data"] = NDArray(Shape(batch_size, 3, width, height), Context::gpu());
  args_map["data_label"] = NDArray(Shape(batch_size), Context::gpu());
  inception_bn_net.InferArgsMap(Context::gpu(), &args_map, args_map);
  exec = inception_bn_net.SimpleBind(Context::gpu(), args_map);
}

void ImageClassify::loadModel(char * model_path) {
  NDArray::Load(std::string(model_path), nullptr, &args_map);
}

std::vector<float> ImageClassify::train(float * data, float * label, bool is_train) {
  NDArray data_n = NDArray(data, Shape(batch_size, 3, width, height), Context::gpu());

  NDArray label_n = NDArray(label, Shape(batch_size), Context::gpu());

  data_n.CopyTo(&args_map["data"]);
  label_n.CopyTo(&args_map["data_label"]);

  std::vector<mx_float> label_data(batch_size * 3 * width * height);
  args_map["data"].SyncCopyToCPU(&label_data, batch_size * 3 * width * height);

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
