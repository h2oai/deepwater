/*!
 * Copyright (c) 2016 by Contributors
 */
#include <cmath>
#include <string>
#include <vector>
#include "network_def.hpp"

using namespace mxnet::cpp;

Symbol AlexnetSymbol(int num_classes) {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
  Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b,
                             Shape(11, 11), 96, Shape(4, 4));
  Symbol relu1 = Activation("relu1", conv1, "relu");
  Symbol pool1 = Pooling("pool1", relu1, Shape(3, 3),
                         PoolingPoolType::max, false, Shape(2, 2));
  Symbol lrn1 = LRN("lrn1", pool1, 5, 0.0001, 0.75, 1);

  Symbol conv2_w("conv2_w"), conv2_b("conv2_b");
  Symbol conv2 = Convolution("conv2", lrn1, conv2_w, conv2_b,
                             Shape(5, 5), 256,
                             Shape(1, 1), Shape(1, 1), Shape(2, 2));
  Symbol relu2 = Activation("relu2", conv2, "relu");
  Symbol pool2 = Pooling("pool2", relu2, Shape(3, 3),
                         PoolingPoolType::max, false, Shape(2, 2));
  Symbol lrn2 = LRN("lrn2", pool2, 5, 0.0001, 0.75, 1);

  Symbol conv3_w("conv3_w"), conv3_b("conv3_b");
  Symbol conv3 = Convolution("conv3", lrn2, conv3_w, conv3_b,
                             Shape(3, 3), 384,
                             Shape(1, 1), Shape(1, 1), Shape(1, 1));
  Symbol relu3 = Activation("relu3", conv3, "relu");

  Symbol conv4_w("conv4_w"), conv4_b("conv4_b");
  Symbol conv4 = Convolution("conv4", relu3, conv4_w, conv4_b,
                             Shape(3, 3), 384,
                             Shape(1, 1), Shape(1, 1), Shape(1, 1));
  Symbol relu4 = Activation("relu4", conv4, "relu");

  Symbol conv5_w("conv5_w"), conv5_b("conv5_b");
  Symbol conv5 = Convolution("conv5", relu4, conv5_w, conv5_b,
                             Shape(3, 3), 256,
                             Shape(1, 1), Shape(1, 1), Shape(1, 1));
  Symbol relu5 = Activation("relu5", conv5, "relu");
  Symbol pool3 = Pooling("pool3", relu5, Shape(3, 3),
                         PoolingPoolType::max, false, Shape(2, 2));

  Symbol flatten = Flatten("flatten", pool3);
  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 4096);
  Symbol relu6 = Activation("relu6", fc1, "relu");
  Symbol dropout1 = Dropout("dropout1", relu6, 0.5);

  Symbol fc2_w("fc2_w"), fc2_b("fc2_b");
  Symbol fc2 = FullyConnected("fc2", dropout1, fc2_w, fc2_b, 4096);
  Symbol relu7 = Activation("relu7", fc2, "relu");
  Symbol dropout2 = Dropout("dropout2", relu7, 0.5);

  Symbol fc3_w("fc3_w"), fc3_b("fc3_b");
  Symbol fc3 = FullyConnected("fc3", dropout2, fc3_w, fc3_b, num_classes);

  return SoftmaxOutput("softmax", fc3, data_label);
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

  Symbol conv1 = ConvFactory(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));
  Symbol conv2 = ConvFactory(pool1, 64, Shape(1, 1), Shape(1, 1),
                             Shape(0, 0), "conv2");
  Symbol conv3 = ConvFactory(conv2, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv3");
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
                         false, Shape(1, 1));
  Symbol flatten = Flatten("flatten", pool6);

  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, num_classes);

  return SoftmaxOutput("softmax", fc1, data_label);
}

Symbol ConvFactory(Symbol data, int num_filter,
                   Shape kernel, Shape stride, Shape pad,
                   const std::string & name,
                   const std::string & suffix) {
  Symbol conv_w("conv_" + name + suffix + "_w"), conv_b("conv_" + name + suffix + "_b");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  return Activation("relu_" + name + suffix, conv, "relu");
}


Symbol ConvFactoryBN(Symbol data, int num_filter,
                     Shape kernel, Shape stride, Shape pad,
                     const std::string & name,
                     const std::string & suffix) {
  Symbol conv_w("conv_" + name + suffix + "_w"), conv_b("conv_" + name + suffix + "_b");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  Symbol bn = BatchNorm("bn_" + name + suffix, conv);
  return Activation("relu_" + name + suffix, bn, "relu");
}

Symbol InceptionFactoryA(Symbol data, int num_1x1, int num_3x3red,
                         int num_3x3, int num_d3x3red, int num_d3x3,
                         PoolingPoolType pool, int proj,
                         const std::string & name) {
  Symbol c1x1 = ConvFactoryBN(data, num_1x1, Shape(1, 1), Shape(1, 1),
                              Shape(0, 0), name + "1x1");
  Symbol c3x3r = ConvFactoryBN(data, num_3x3red, Shape(1, 1), Shape(1, 1),
                               Shape(0, 0), name + "_3x3r");
  Symbol c3x3 = ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(1, 1),
                              Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = ConvFactoryBN(data = cd3x3, num_d3x3, Shape(3, 3), Shape(1, 1),
                        Shape(1, 1), name + "_double_3x3_1");
  Symbol pooling = Pooling(name + "_pool", data,
                           Shape(3, 3), pool, false,
                           Shape(1, 1), Shape(1, 1));
  Symbol cproj = ConvFactoryBN(pooling, proj, Shape(1, 1), Shape(1, 1),
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
  Symbol c3x3r = ConvFactoryBN(data, num_3x3red, Shape(1, 1),
                               Shape(1, 1), Shape(0, 0),
                               name + "_3x3", "_reduce");
  Symbol c3x3 = ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(2, 2),
                              Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = ConvFactoryBN(cd3x3, num_d3x3, Shape(3, 3), Shape(2, 2),
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
  Symbol conv1 = ConvFactoryBN(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::max, false, Shape(2, 2));

  // stage 2
  Symbol conv2red = ConvFactoryBN(pool1, 64, Shape(1, 1), Shape(1, 1),  Shape(0, 0), "conv2red");
  Symbol conv2 = ConvFactoryBN(conv2red, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv2");
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

Symbol VGGSymbol(int num_classes) {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol conv1_1_w("conv1_1_w"), conv1_1_b("conv1_1_b");
  Symbol conv1_1 = Convolution("conv1_1", data, conv1_1_w, conv1_1_b,
                               Shape(3, 3), 64, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu1_1 = Activation("relu1_1", conv1_1, "relu");
  Symbol pool1 = Pooling("pool1", relu1_1, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv2_1_w("conv2_1_w"), conv2_1_b("conv2_1_b");
  Symbol conv2_1 = Convolution("conv2_1", pool1, conv2_1_w, conv2_1_b,
                               Shape(3, 3), 128, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu2_1 = Activation("relu2_1", conv2_1, "relu");
  Symbol pool2 = Pooling("pool2", relu2_1, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv3_1_w("conv3_1_w"), conv3_1_b("conv3_1_b");
  Symbol conv3_1 = Convolution("conv3_1", pool2, conv3_1_w, conv3_1_b,
                               Shape(3, 3), 256, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu3_1 = Activation("relu3_1", conv3_1, "relu");
  Symbol conv3_2_w("conv3_2_w"), conv3_2_b("conv3_2_b");
  Symbol conv3_2 = Convolution("conv3_2", relu3_1, conv3_2_w, conv3_2_b,
                               Shape(3, 3), 256, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu3_2 = Activation("relu3_2", conv3_2, "relu");
  Symbol pool3 = Pooling("pool3", relu3_2, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv4_1_w("conv4_1_w"), conv4_1_b("conv4_1_b");
  Symbol conv4_1 = Convolution("conv4_1", pool3, conv4_1_w, conv4_1_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu4_1 = Activation("relu4_1", conv4_1, "relu");

  Symbol conv4_2_w("conv4_2_w"), conv4_2_b("conv4_2_b");
  Symbol conv4_2 = Convolution("conv4_2", relu4_1, conv4_2_w, conv4_2_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu4_2 = Activation("relu4_2", conv4_2, "relu");
  Symbol pool4 = Pooling("pool4", relu4_2, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv5_1_w("conv5_1_w"), conv5_1_b("conv5_1_b");
  Symbol conv5_1 = Convolution("conv5_1", pool4, conv5_1_w, conv5_1_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu5_1 = Activation("relu5_1", conv5_1, "relu");

  Symbol conv5_2_w("conv5_2_w"), conv5_2_b("conv5_2_b");
  Symbol conv5_2 = Convolution("conv5_2", relu5_1, conv5_2_w, conv5_2_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu5_2 = Activation("relu5_2", conv5_2, "relu");
  Symbol pool5 = Pooling("pool5", relu5_2, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol flatten = Flatten("flatten", pool5);
  Symbol fc6_w("fc6_w"), fc6_b("fc6_b");
  Symbol fc6 = FullyConnected("fc6", flatten, fc6_w, fc6_b, 4096);
  Symbol relu6 = Activation("relu6", fc6, "relu");
  Symbol drop6 = Dropout("drop6", relu6, 0.5);

  Symbol fc7_w("fc7_w"), fc7_b("fc7_b");
  Symbol fc7 = FullyConnected("fc7", drop6, fc7_w, fc7_b, 4096);
  Symbol relu7 = Activation("relu7", fc7, "relu");
  Symbol drop7 = Dropout("drop7", relu7, 0.5);

  Symbol fc8_w("fc8_w"), fc8_b("fc8_b");
  Symbol fc8 = FullyConnected("fc8", drop7, fc8_w, fc8_b, num_classes);
  return SoftmaxOutput("softmax", fc8, data_label);
}

Symbol LenetSymbol(int num_classes) {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");
  Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
  Symbol conv2_w("conv2_w"), conv2_b("conv2_b");
  Symbol conv3_w("conv3_w"), conv3_b("conv3_b");
  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc2_w("fc2_w"), fc2_b("fc2_b");

  Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b, Shape(5, 5), 20);
  Symbol tanh1 = Activation("tanh1", conv1, "tanh");
  Symbol pool1 = Pooling("pool1", tanh1, Shape(2, 2), PoolingPoolType::max, false, Shape(2, 2));

  Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b, Shape(5, 5), 50);
  Symbol tanh2 = Activation("tanh2", conv2, "tanh");
  Symbol pool2 = Pooling("pool2", tanh2, Shape(2, 2), PoolingPoolType::max, false, Shape(2, 2));

  Symbol conv3 = Convolution("conv3", pool2, conv3_w, conv3_b, Shape(2, 2), 500);
  Symbol tanh3 = Activation("tanh3", conv3, "tanh");
  Symbol pool3 = Pooling("pool3", tanh3, Shape(2, 2), PoolingPoolType::max, false, Shape(1, 1));

  Symbol flatten = Flatten("flatten", pool3);
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 500);
  Symbol tanh4 = Activation("tanh4", fc1, "tanh");
  Symbol fc2 = FullyConnected("fc2", tanh4, fc2_w, fc2_b, num_classes);

  Symbol lenet = SoftmaxOutput("softmax", fc2, data_label);

  return lenet;
}


Symbol getConv(const std::string & name, Symbol data,
               int  num_filter,
               Shape kernel, Shape stride, Shape pad,
               bool with_relu,
               mx_float bn_momentum) {
  Symbol conv_w(name + "_w");
  Symbol conv = ConvolutionNoBias(name, data, conv_w,
                                  kernel, num_filter, stride, Shape(1, 1),
                                  pad, 1, 512);

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
    Symbol shortcut_w(name + "_proj_w");
    shortcut = ConvolutionNoBias(name + "_proj", data, shortcut_w,
                                 Shape(2, 2), num_filter,
                                 Shape(2, 2), Shape(1, 1), Shape(0, 0),
                                 1, 512);
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

Symbol ResNetSymbol(int num_class, int num_level, int num_block,
                    int num_filter, mx_float bn_momentum,
                    Shape pool_kernel) {
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

std::vector<Symbol> lstm(int num_hidden, Symbol indata,
                         std::vector<Symbol> prev_state, std::vector<Symbol> param,
                         int seqidx, int layeridx, mx_float dropout) {
  if (dropout > 0) indata = Dropout("dp", indata, dropout);

  Symbol i2h =
      FullyConnected("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_i2h",
                     indata, param[0], param[1],
                     num_hidden *4);

  Symbol h2h =
      FullyConnected("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_h2h",
                     prev_state[1], param[2], param[3],
                     num_hidden * 4);

  Symbol gates = i2h + h2h;
  Symbol slice_gates =
      SliceChannel("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_slice",
                   gates, 4);

  Symbol in_gate =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_in_gates",
                 slice_gates[0], "sigmoid");
  Symbol in_transform =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_in_transform",
                 slice_gates[1], "tanh");
  Symbol forget_gate =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_forget_data",
                 slice_gates[2], "sigmoid");
  Symbol out_gate =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_out_gate",
                 slice_gates[3], "sigmoid");

  Symbol next_c = (forget_gate * prev_state[0]) + (in_gate * in_transform);
  Symbol next_h = out_gate * Activation("", next_c, "tanh");
  std::vector<Symbol> state;
  state.push_back(next_c);
  state.push_back(next_h);
  return state;
}

Symbol lstm_unroll(int num_lstm_layer, int seq_len, int input_size,
                   int num_hidden, int num_embed, int num_label,
                   mx_float dropout) {
  Symbol embed_weight = Symbol::Variable("embed_weight");
  Symbol cls_weight = Symbol::Variable("cls_weight");
  Symbol cls_bias = Symbol::Variable("cls_bias");

  std::vector<std::vector<Symbol>> param_cells;
  std::vector<std::vector<Symbol>> last_states;

  for (int i = 0; i < num_lstm_layer; i++) {
    std::vector<Symbol> param;
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_i2h_weight"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_i2h_bias"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_h2h_weight"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_h2h_bias"));
    param_cells.push_back(param);

    std::vector<Symbol> state;
    state.push_back(Symbol::Variable("l" + std::to_string(i) + "_init_c"));
    state.push_back(Symbol::Variable("l" + std::to_string(i) + "_init_h"));
    last_states.push_back(state);
  }

  Symbol label = Symbol::Variable("label");
  std::vector<Symbol> last_hidden;
  for (int seqidx = 0; seqidx < seq_len; seqidx++) {
    Symbol data = Symbol::Variable("t" + std::to_string(seqidx) + "_data");
    Symbol hidden = Embedding("t" + std::to_string(seqidx) + "_embed",
                              data, embed_weight, input_size, num_embed);
    for (int i = 0; i < num_lstm_layer; i++) {
      mx_float dp;
      if (i ==0)
        dp = 0.0;
      else
        dp = dropout;

      std::vector<Symbol> next_state = lstm(num_hidden, hidden, last_states[i],
                                            param_cells[i], seqidx, i, dp);
      hidden = next_state[1];
      last_states[i] = next_state;
    }

    if (dropout > 0)
      hidden = Dropout("", hidden, dropout);

    last_hidden.push_back(hidden);
  }

  Symbol concat = Concat("", last_hidden, last_hidden.size(), 0);
  Symbol fc = FullyConnected("", concat, cls_weight, cls_bias, num_label);
  Symbol sm = SoftmaxOutput("sm", fc, label);
  std::vector<Symbol> list_all;
  list_all.push_back(sm);
  for (int i = 0; i < num_lstm_layer; i++) {
    last_states[i][0] = BlockGrad("l" + std::to_string(i) + "_last_c", last_states[i][0]);
    last_states[i][1] = BlockGrad("l" + std::to_string(i) + "_last_c", last_states[i][1]);
  }

  for (size_t i = 0; i < last_states.size(); i++) {
    list_all.push_back(last_states[i][0]);
  }

  for (size_t i = 0; i < last_states.size(); i++) {
    list_all.push_back(last_states[i][1]);
  }

  return Symbol::Group(list_all);
}
