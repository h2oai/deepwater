#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "MxNetCpp.h"

using namespace mxnet::cpp;

Symbol ConvFactory(Symbol data, int num_filter,
                   Shape kernel, Shape stride = Shape(1, 1),
                   Shape pad = Shape(0, 0),
                   const std::string & name = "",
                   const std::string & suffix = "") {
  Symbol conv_w("conv_" + name + suffix + "_w"), conv_b("conv_" + name + suffix + "_b");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1,1), pad);
  Symbol bn = BatchNorm("bn_" + name + suffix, conv, 0.001, 0.9, false);
  return Activation("relu_" + name + suffix, bn, "relu");
}

Symbol InceptionFactoryA(Symbol data, int num_1x1, int num_3x3red,
                         int num_3x3, int num_d3x3red, int num_d3x3,
                         PoolingPoolType pool, int proj,
                         const std::string & name) {

  Symbol c1x1 = ConvFactory(data, num_1x1, Shape(1,1), Shape(1,1),
                            Shape(0,0), name + "1x1");
  Symbol c3x3r = ConvFactory(data, num_3x3red, Shape(1,1), Shape(1,1),
                             Shape(0,0), name + "_3x3");
  Symbol c3x3 = ConvFactory(c3x3r, num_3x3, Shape(3, 3), Shape(1,1),
                            Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = ConvFactory(data, num_d3x3red, Shape(1, 1), Shape(1, 1), 
                              Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = ConvFactory(cd3x3r, num_d3x3, Shape(3, 3), Shape(1,1),
                             Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = ConvFactory(data=cd3x3, num_d3x3, Shape(3, 3), Shape(1, 1), 
                      Shape(1, 1), name + "_double_3x3_1");
  Symbol pooling = Pooling(name + "_pool", data,
                           Shape(3, 3), pool, false,
                           Shape(1,1), Shape(1,1));
  Symbol cproj = ConvFactory(pooling, proj, Shape(1, 1), Shape(1, 1),
                             Shape(0, 0), name + "_proj");
  std::vector<Symbol> lst;
  lst.push_back(c1x1);
  lst.push_back(c3x3);
  lst.push_back(cd3x3);
  lst.push_back(cproj); 
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol InceptionFactoryB(Symbol data, int num_3x3red, int num_3x3,
                         int num_d3x3red, int num_d3x3, const std::string & name) {
  Symbol c3x3r = ConvFactory(data, num_3x3red, Shape(1, 1), 
                             Shape(1, 1), Shape(0, 0),
                             name + "_3x3", "_reduce");
  Symbol c3x3 = ConvFactory(c3x3r, num_3x3, Shape(3, 3), Shape(2, 2),
                            Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = ConvFactory(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                              Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = ConvFactory(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                             Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = ConvFactory(cd3x3, num_d3x3, Shape(3, 3), Shape(2, 2),
                      Shape(1, 1), name + "_double_3x3_1");
  Symbol pooling = Pooling("max_pool_" + name + "_pool", data,
                           Shape(3, 3), PoolingPoolType::max,
                           false, Shape(2, 2));
  std::vector<Symbol> lst;
  lst.push_back(c3x3);
  lst.push_back(cd3x3);
  lst.push_back(pooling);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol InceptionSymbol(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  // stage 1
  Symbol conv1 = ConvFactory(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::max, false, Shape(2, 2));

  // stage 2
  Symbol conv2red = ConvFactory(pool1, 64, Shape(1, 1), Shape(1, 1),  Shape(0, 0), "conv2red");
  Symbol conv2 = ConvFactory(conv2red, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv2");
  Symbol pool2 = Pooling("pool2", conv2, Shape(3, 3), PoolingPoolType::max, false, Shape(2, 2));

  // stage 3
  Symbol in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, PoolingPoolType::avg, 32, "3a");
  Symbol in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, PoolingPoolType::avg, 64, "3b");
  Symbol in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, "3c");

  // stage 4
  Symbol in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, PoolingPoolType::avg, 128, "4a");
  Symbol in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128,  PoolingPoolType::avg, 128, "4b");
  Symbol in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, PoolingPoolType::avg, 128, "4c");
  Symbol in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192,  PoolingPoolType::avg, 128, "4d");
  Symbol in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, "4e");

  // stage 5
  Symbol in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, PoolingPoolType::avg, 128, "5a");
  Symbol in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, PoolingPoolType::max, 128, "5b");

  // average pooling
  Symbol avg = Pooling("global_pool", in5b, Shape(7, 7), PoolingPoolType::avg);

  // classifier
  Symbol flatten = Flatten("flatten", avg);
  Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, conv1_w, conv1_b, num_classes);
  return SoftmaxOutput("softmax", fc1, data_label);
}

int main(int argc, char const *argv[]) {


}
