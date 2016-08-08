/*!
 * Copyright (c) 2016 by Contributors
 */

#include "network_def.hpp"

using namespace mxnet::cpp;

Symbol MLPSymbol(int num_classes) {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");
  Symbol fc1_w("fc1_weight"), fc1_b("fc1_bias");
  Symbol fc1 = FullyConnected("fc1", data, fc1_w, fc1_b, 128);
  Symbol act1 = Activation("relu1", fc1, "relu");
  Symbol fc2_w("fc2_weight"), fc2_b("fc2_bias");
  Symbol fc2 = FullyConnected("fc2", act1, fc2_w, fc2_b, 64);
  Symbol act2 = Activation("relu2", fc2, "relu");
  Symbol fc3_w("fc3_weight"), fc3_b("fc3_bias");
  Symbol fc3 = FullyConnected("fc3", act2, fc3_w, fc3_b, num_classes);
  return SoftmaxOutput("softmax", fc3, data_label);
}

int main() {

  Symbol alexnet = AlexnetSymbol(10);
  alexnet.Save("./test/symbol_alexnet-cpp.json");

  Symbol googlenet = GoogleNetSymbol(10);
  googlenet.Save("./test/symbol_googlenet-cpp.json");

  Symbol inception = InceptionSymbol(10);
  inception.Save("./test/symbol_inception-bn-cpp.json");

  Symbol mlp = MLPSymbol(10);
  mlp.Save("./test/symbol_mlp-cpp.json");

  Symbol lenet = LenetSymbol(10);
  lenet.Save("./test/symbol_lenet-cpp.json");

  Symbol vgg = VGGSymbol(10);
  vgg.Save("./test/symbol_vgg-cpp.json");

  Symbol resnet = ResNetSymbol(10);
  resnet.Save("./test/symbol_resnet-cpp.json");

  return 0;
}
