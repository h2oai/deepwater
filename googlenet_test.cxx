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

Symbol InceptionFactory(Symbol data, int num_1x1, int num_3x3red, 
                        int num_3x3, int num_d5x5red, int num_d5x5,
                        PoolingPoolType pool, int proj, const std::string & name) {
  Symbol c1x1 = ConvFactory(data, num_1x1, Shape(1, 1), 
                            Shape(1, 1), Shape(0, 0), name + "_1x1");

  Symbol c3x3r = ConvFactory(data, num_3x3red, Shape(1, 1), 
                             Shape(1, 1), Shape(0, 0), name + "_3x3", "_reduce");
  Symbol c3x3 = ConvFactory(c3x3r, num_3x3, Shape(3, 3), 
                            Shape(1, 1), Shape(1, 1), name + "_3x3");

  Symbol cd5x5r = ConvFactory(data, num_d5x5red, Shape(1, 1), 
                              Shape(1, 1), Shape(0, 0), name + "_5x5", "_reduce");
  Symbol cd5x5 = ConvFactory(cd5x5r, num_d5x5, Shape(5, 5), 
                             Shape(1, 1), Shape(2, 2), name + "_5x5");

  Symbol pooling = Pooling(name + "_pool", data, Shape(3, 3), pool,
                           false, Shape(1, 1), Shape(1, 1));
  Symbol cproj = ConvFactory(pooling, proj, Shape(1, 1), 
                             Shape(1, 1), Shape(0, 0), name + "_proj");

  std::vector<Symbol> lst;
  lst.push_back(c1x1);
  lst.push_back(c3x3);
  lst.push_back(cd5x5);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol GoogleNetSymbol(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol conv1 = ConvFactory(data, 64, Shape(7, 7), Shape(2,2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::max, 
                         false, Shape(2, 2));
  Symbol conv2 = ConvFactory(pool1, 64, Shape(1, 1), Shape(1,1), 
                             Shape(0, 0), "conv2");
  Symbol conv3 = ConvFactory(conv2, 192, Shape(3, 3), Shape(1, 1), Shape(1,1), "conv3");
  Symbol pool3 = Pooling("pool3", conv3, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, PoolingPoolType::max, 32, "in3a");
  Symbol in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, PoolingPoolType::max, 64, "in3b");
  Symbol pool4 = Pooling("pool4", in3b, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));
  Symbol in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, PoolingPoolType::max, 64, "in4a");
  Symbol in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, PoolingPoolType::max, 64, "in4b");
  Symbol in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, PoolingPoolType::max, 64, "in4c");
  Symbol in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, PoolingPoolType::max, 64, "in4d");
  Symbol in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, PoolingPoolType::max, 128, "in4e");
  Symbol pool5 = Pooling("pool5", in4e, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));
  Symbol in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, PoolingPoolType::max, 128, "in5a");
  Symbol in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, PoolingPoolType::max, 128, "in5b");
  Symbol pool6 = Pooling("pool6", in5b, Shape(7, 7), PoolingPoolType::avg, 
                         false, Shape(1,1));
  Symbol flatten = Flatten("flatten", pool6);

  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, num_classes);

  return SoftmaxOutput("softmax", fc1, data_label);
}

int main(int argc, char const *argv[]) {


}
