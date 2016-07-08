#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "MxNetCpp.h"

using namespace mxnet::cpp;

Symbol getConv(const std::string & name, Symbol data,
               int  num_filter,
               Shape kernel, Shape stride, Shape pad,
               bool with_relu,
               mx_float bn_momentum) {

  Symbol conv_w("conv_w"), conv_b("conv_b");
  Symbol conv = Convolution(name, data, conv_w, conv_b,
                            kernel, num_filter, stride, Shape(1, 1),
                            pad, 1, 512, true);

  Symbol bn = BatchNorm(name + "_bn", conv, 2e-5, bn_momentum, false);

  if (with_relu) {
    return Activation(name + "_relu", bn, "relu");
  } else {
    return bn;
  }

}

Symbol makeBlock(const std::string & name, Symbol data, int num_filter, 
                 bool dim_match, mx_float bn_momentum) {
  Shape stride;
  if (dim_match) {
    stride = Shape(1, 1);
  } else {
    stride = Shape(2, 2);
  }

  Symbol conv1 = getConv(name + "_conv1", data, num_filter,
                         Shape(3, 3), stride, Shape(1, 1), 
                         true, bn_momentum);

  Symbol conv2 = getConv(name + "_conv2", conv1, num_filter,
                         Shape(3, 3), Shape(1, 1), Shape(1, 1),
                         false, bn_momentum);

  Symbol shortcut;

  if (dim_match) {
    shortcut = data;
  } else {
    Symbol shortcut_w(name + "_proj_w"), shortcut_b(name + "_proj_b");
    shortcut = Convolution(name + "_proj", data, shortcut_w, shortcut_b,
                           Shape(2, 2), num_filter,
                           Shape(2, 2), Shape(1, 1), Shape(0, 0),
                           1, 512, true);
  }

  Symbol fused = shortcut + conv2;
  return Activation(name + "_relu", fused, "relu");

}

Symbol getBody(Symbol data, int num_level, int num_block, int num_filter, mx_float bn_momentum) {
  for (int level = 0; level < num_level; level++) {
    for (int block = 0; block < num_block; block++) {
      data = makeBlock("level" + std::to_string(level + 1) + "_block" + std::to_string(block + 1),
                       data, num_filter * (std::pow(2, level)),
                       (level == 0 || block > 0), bn_momentum);
    }
  }
  return data;
}

Symbol InceptionSymbol(int num_class, int num_level = 3, int num_block = 9,
                       int num_filter = 16, mx_float bn_momentum = 0.9,
                       Shape pool_kernel = Shape(8, 8)) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol zscore = BatchNorm("zscore", data, 0.001, bn_momentum);

  Symbol conv = getConv("conv0", zscore, num_filter, 
                        Shape(3, 3), Shape(1, 1), Shape(1, 1),
                        true, bn_momentum);

  Symbol body = getBody(conv, num_level, num_block, num_filter, bn_momentum);

  Symbol pool = Pooling("pool", body, pool_kernel, PoolingPoolType::avg);

  Symbol flat = Flatten("flatten", pool);

  Symbol fc_w("fc_w"), fc_b("fc_b");
  Symbol fc = FullyConnected("fc", flat, fc_w, fc_b, num_class);

  return SoftmaxOutput("softmax", fc, data_label);
}

int main(int argc, char const *argv[]) {


}
