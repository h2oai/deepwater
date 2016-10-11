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
  Symbol data_label = Symbol::Variable("softmax_label");

  Symbol conv1_w("convolution0_weight"), conv1_b("convolution0_bias");
  Symbol conv1 = Convolution("convolution0", data, conv1_w, conv1_b,
                             Shape(11, 11), 96,
                             Shape(4, 4), Shape(1, 1), Shape(2, 2));
  Symbol relu1 = Activation("activation0", conv1, "relu");
  Symbol pool1 = Pooling("pooling0", relu1, Shape(3, 3),
                         PoolingPoolType::max, false, Shape(2, 2));
  Symbol lrn1 = LRN("lrn0", pool1, 5, 0.0001, 0.75, 1);

  Symbol conv2_w("convolution1_weight"), conv2_b("convolution1_bias");
  Symbol conv2 = Convolution("convolution1", lrn1, conv2_w, conv2_b,
                             Shape(5, 5), 256,
                             Shape(1, 1), Shape(1, 1), Shape(2, 2));
  Symbol relu2 = Activation("activation1", conv2, "relu");
  Symbol pool2 = Pooling("pooling1", relu2, Shape(3, 3),
                         PoolingPoolType::max, false, Shape(2, 2));
  Symbol lrn2 = LRN("lrn1", pool2, 5, 0.0001, 0.75, 1);

  Symbol conv3_w("convolution2_weight"), conv3_b("convolution2_bias");
  Symbol conv3 = Convolution("convolution2", lrn2, conv3_w, conv3_b,
                             Shape(3, 3), 384,
                             Shape(1, 1), Shape(1, 1), Shape(1, 1));
  Symbol relu3 = Activation("activation2", conv3, "relu");

  Symbol conv4_w("convolution3_weight"), conv4_b("convolution3_bias");
  Symbol conv4 = Convolution("convolution3", relu3, conv4_w, conv4_b,
                             Shape(3, 3), 384,
                             Shape(1, 1), Shape(1, 1), Shape(1, 1));
  Symbol relu4 = Activation("activation3", conv4, "relu");

  Symbol conv5_w("convolution4_weight"), conv5_b("convolution4_bias");
  Symbol conv5 = Convolution("convolution4", relu4, conv5_w, conv5_b,
                             Shape(3, 3), 256,
                             Shape(1, 1), Shape(1, 1), Shape(1, 1));

  Symbol relu5 = Activation("activation4", conv5, "relu");

  Symbol pool3 = Pooling("pooling2", relu5, Shape(3, 3),
                         PoolingPoolType::max, false, Shape(2, 2));

  Symbol flatten = Flatten("flatten0", pool3);
  Symbol fc1_w("fullyconnected0_weight"), fc1_b("fullyconnected0_bias");
  Symbol fc1 = FullyConnected("fullyconnected0", flatten, fc1_w, fc1_b, 4096);
  Symbol relu6 = Activation("activation5", fc1, "relu");
  Symbol dropout1 = Dropout("dropout0", relu6, 0.5);

  Symbol fc2_w("fullyconnected1_weight"), fc2_b("fullyconnected1_bias");
  Symbol fc2 = FullyConnected("fullyconnected1", dropout1, fc2_w, fc2_b, 4096);
  Symbol relu7 = Activation("activation6", fc2, "relu");
  Symbol dropout2 = Dropout("dropout1", relu7, 0.5);

  Symbol fc3_w("fullyconnected2_weight"), fc3_b("fullyconnected2_bias");
  Symbol fc3 = FullyConnected("fullyconnected2", dropout2, fc3_w, fc3_b, num_classes);

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

  Symbol pooling =
      Pooling(PoolingPoolTypeValues[static_cast<int>(pool)] + "_pool_" +name + "_pool",
              data, Shape(3, 3), pool,
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
  Symbol data_label = Symbol::Variable("softmax_label");

  Symbol conv1 = ConvFactory(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pooling0", conv1, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));
  Symbol conv2 = ConvFactory(pool1, 64, Shape(1, 1), Shape(1, 1),
                             Shape(0, 0), "conv2");
  Symbol conv3 = ConvFactory(conv2, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv3");
  Symbol pool3 = Pooling("pooling1", conv3, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, PoolingPoolType::max, 32, "in3a");
  Symbol in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, PoolingPoolType::max, 64, "in3b");
  Symbol pool4 = Pooling("pooling2", in3b, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));
  Symbol in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, PoolingPoolType::max, 64, "in4a");
  Symbol in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, PoolingPoolType::max, 64, "in4b");
  Symbol in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, PoolingPoolType::max, 64, "in4c");
  Symbol in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, PoolingPoolType::max, 64, "in4d");
  Symbol in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, PoolingPoolType::max, 128, "in4e");
  Symbol pool5 = Pooling("pooling3", in4e, Shape(3, 3), PoolingPoolType::max,
                         false, Shape(2, 2));
  Symbol in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, PoolingPoolType::max, 128, "in5a");
  Symbol in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, PoolingPoolType::max, 128, "in5b");
  Symbol pool6 = Pooling("pooling4", in5b, Shape(7, 7), PoolingPoolType::avg,
                         false, Shape(1, 1));
  Symbol flatten = Flatten("flatten0", pool6);

  Symbol fc1_w("fullyconnected0_weight"), fc1_b("fullyconnected0_bias");
  Symbol fc1 = FullyConnected("fullyconnected0", flatten, fc1_w, fc1_b, num_classes);

  return SoftmaxOutput("softmax", fc1, data_label);
}

Symbol ConvFactory(Symbol data, int num_filter,
                   Shape kernel, Shape stride, Shape pad,
                   const std::string & name,
                   const std::string & suffix) {
  Symbol conv_w("conv_" + name + suffix + "_weight"), conv_b("conv_" + name + suffix + "_bias");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  return Activation("relu_" + name + suffix, conv, "relu");
}

Symbol ConvFactoryBN(Symbol data, int num_filter,
                     Shape kernel, Shape stride, Shape pad,
                     const std::string & name,
                     const std::string & suffix,
                     mx_float eps, mx_float momentum) {
  Symbol conv_w("conv_" + name + suffix + "_weight"), conv_b("conv_" + name + suffix + "_bias");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  Symbol bn = BatchNorm("bn_" + name + suffix, conv, eps, momentum);
  return Activation("relu_" + name + suffix, bn, "relu");
}

Symbol ConvFactoryNoBias(Symbol data, int num_filter,
                         Shape kernel, Shape stride, Shape pad,
                         const std::string & name,
                         const std::string & suffix) {
  Symbol conv_w("conv_" + name + suffix + "_weight");

  Symbol conv = ConvolutionNoBias("conv_" + name + suffix, data,
                                  conv_w, kernel,
                                  num_filter, stride, Shape(1, 1), pad);
  Symbol bn = BatchNorm("bn_" + name + suffix, conv);
  return Activation("relu_" + name + suffix, bn, "relu");
}

Symbol InceptionFactoryA(Symbol data, int num_1x1, int num_3x3red,
                         int num_3x3, int num_d3x3red, int num_d3x3,
                         PoolingPoolType pool, int proj,
                         const std::string & name,
                         mx_float eps, mx_float momentum) {
  Symbol c1x1 = ConvFactoryBN(data, num_1x1, Shape(1, 1), Shape(1, 1),
                              Shape(0, 0), name + "_1x1", "", eps, momentum);
  Symbol c3x3r = ConvFactoryBN(data, num_3x3red, Shape(1, 1), Shape(1, 1),
                               Shape(0, 0), name + "_3x3", "_reduce", eps, momentum);
  Symbol c3x3 = ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(1, 1),
                              Shape(1, 1), name + "_3x3", "", eps, momentum);
  Symbol cd3x3r = ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce", eps, momentum);
  Symbol cd3x3 = ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0", "", eps, momentum);

  cd3x3 = ConvFactoryBN(cd3x3, num_d3x3, Shape(3, 3), Shape(1, 1),
                        Shape(1, 1), name + "_double_3x3_1", "", eps, momentum);
  Symbol pooling =
      Pooling(PoolingPoolTypeValues[static_cast<int>(pool)] + "_pool_" +name + "_pool", data,
              Shape(3, 3), pool, false,
              Shape(1, 1), Shape(1, 1));
  Symbol cproj = ConvFactoryBN(pooling, proj, Shape(1, 1), Shape(1, 1),
                               Shape(0, 0), name + "_proj", "", eps, momentum);
  std::vector<Symbol> lst;
  lst.push_back(c1x1);
  lst.push_back(c3x3);
  lst.push_back(cd3x3);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol InceptionFactoryB(Symbol data, int num_3x3red, int num_3x3,
                         int num_d3x3red, int num_d3x3, const std::string & name,
                         mx_float eps, mx_float momentum) {
  Symbol c3x3r = ConvFactoryBN(data, num_3x3red, Shape(1, 1),
                               Shape(1, 1), Shape(0, 0),
                               name + "_3x3", "_reduce", eps, momentum);
  Symbol c3x3 = ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(2, 2),
                              Shape(1, 1), name + "_3x3", "", eps, momentum);
  Symbol cd3x3r = ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce", eps, momentum);
  Symbol cd3x3 = ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0", "", eps, momentum);
  cd3x3 = ConvFactoryBN(cd3x3, num_d3x3, Shape(3, 3), Shape(2, 2),
                        Shape(1, 1), name + "_double_3x3_1", "", eps, momentum);
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
  Symbol data_label = Symbol::Variable("softmax_label");

  // stage 1
  Symbol conv1 = ConvFactoryBN(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::max, false, Shape(2, 2));

  // stage 2
  Symbol conv2red = ConvFactoryBN(pool1, 64, Shape(1, 1), Shape(1, 1),  Shape(0, 0), "conv2red");
  Symbol conv2 = ConvFactoryBN(conv2red, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv2");
  Symbol pool2 =
      Pooling("pool2", conv2, Shape(3, 3), PoolingPoolType::max, false, Shape(2, 2), Shape(0, 0));

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
  Symbol fc1_w("fc1_weight"), fc1_b("fc1_bias");
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, num_classes);
  return SoftmaxOutput("softmax", fc1, data_label);
}

Symbol InceptionSymbol2(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");

  // stage 1
  Symbol conv1 =
      ConvFactoryBN(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "1", "", 1e-10, 0.1);
  Symbol pool1 =
      Pooling("max_pool_1", conv1, Shape(3, 3), PoolingPoolType::max, false, Shape(2, 2));

  // stage 2
  Symbol conv2red_w("conv_2_reduce_weight"), conv2red_b("conv_2_reduce_bias");
  Symbol conv2red = Convolution("conv_2_reduce", pool1,
                                conv2red_w, conv2red_b, Shape(1, 1),
                                64, Shape(1, 1), Shape(1, 1), Shape(0, 0));
  Symbol bn2red = BatchNorm("bn_2_1", conv2red, 1e-10, 0.1);
  Symbol act2red = Activation("relu_2_1", bn2red, "relu");
  Symbol conv2 =
      ConvFactoryBN(act2red, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "2", "", 1e-10, 0.1);

  Symbol pool2 =
      Pooling("max_pool_2", conv2, Shape(3, 3),
              PoolingPoolType::max, false, Shape(2, 2), Shape(0, 0));

  // stage 3
  Symbol in3a =
      InceptionFactoryA(pool2, 64, 64, 64, 64, 96, PoolingPoolType::avg, 32, "3a", 1e-10, 0.1);
  Symbol in3b =
      InceptionFactoryA(in3a, 64, 64, 96, 64, 96, PoolingPoolType::avg, 64, "3b", 1e-10, 0.1);
  Symbol in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, "3c", 1e-10, 0.1);

  // stage 4
  Symbol in4a =
      InceptionFactoryA(in3c, 224, 64, 96, 96, 128, PoolingPoolType::avg, 128, "4a", 1e-10, 0.1);
  Symbol in4b =
      InceptionFactoryA(in4a, 192, 96, 128, 96, 128,  PoolingPoolType::avg, 128, "4b", 1e-10, 0.1);
  Symbol in4c =
      InceptionFactoryA(in4b, 160, 128, 160, 128, 160, PoolingPoolType::avg, 128, "4c", 1e-10, 0.1);
  Symbol in4d =
      InceptionFactoryA(in4c, 96, 128, 192, 160, 192,  PoolingPoolType::avg, 128, "4d", 1e-10, 0.1);
  Symbol in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, "4e", 1e-10, 0.1);

  // stage 5
  Symbol in5a =
      InceptionFactoryA(in4e, 352, 192, 320, 160, 224, PoolingPoolType::avg, 128, "5a", 1e-10, 0.1);
  Symbol in5b =
      InceptionFactoryA(in5a, 352, 192, 320, 192, 224, PoolingPoolType::max, 128, "5b", 1e-10, 0.1);

  // average pooling
  Symbol avg = Pooling("global_pool", in5b, Shape(7, 7), PoolingPoolType::avg);

  // classifier
  Symbol flatten = Flatten("flatten", avg);
  Symbol fc1_w("fc_weight"), fc1_b("fc_bias");
  Symbol fc1 = FullyConnected("fc", flatten, fc1_w, fc1_b, num_classes);
  return SoftmaxOutput("softmax", fc1, data_label);
}

Symbol VGGSymbol(int num_classes) {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");

  Symbol conv1_1_w("conv1_1_weight"), conv1_1_b("conv1_1_bias");
  Symbol conv1_1 = Convolution("conv1_1", data, conv1_1_w, conv1_1_b,
                               Shape(3, 3), 64, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu1_1 = Activation("relu1_1", conv1_1, "relu");
  Symbol pool1 = Pooling("pool1", relu1_1, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv2_1_w("conv2_1_weight"), conv2_1_b("conv2_1_bias");
  Symbol conv2_1 = Convolution("conv2_1", pool1, conv2_1_w, conv2_1_b,
                               Shape(3, 3), 128, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu2_1 = Activation("relu2_1", conv2_1, "relu");
  Symbol pool2 = Pooling("pool2", relu2_1, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv3_1_w("conv3_1_weight"), conv3_1_b("conv3_1_bias");
  Symbol conv3_1 = Convolution("conv3_1", pool2, conv3_1_w, conv3_1_b,
                               Shape(3, 3), 256, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu3_1 = Activation("relu3_1", conv3_1, "relu");
  Symbol conv3_2_w("conv3_2_weight"), conv3_2_b("conv3_2_bias");
  Symbol conv3_2 = Convolution("conv3_2", relu3_1, conv3_2_w, conv3_2_b,
                               Shape(3, 3), 256, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu3_2 = Activation("relu3_2", conv3_2, "relu");
  Symbol pool3 = Pooling("pool3", relu3_2, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv4_1_w("conv4_1_weight"), conv4_1_b("conv4_1_bias");
  Symbol conv4_1 = Convolution("conv4_1", pool3, conv4_1_w, conv4_1_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu4_1 = Activation("relu4_1", conv4_1, "relu");

  Symbol conv4_2_w("conv4_2_weight"), conv4_2_b("conv4_2_bias");
  Symbol conv4_2 = Convolution("conv4_2", relu4_1, conv4_2_w, conv4_2_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu4_2 = Activation("relu4_2", conv4_2, "relu");
  Symbol pool4 = Pooling("pool4", relu4_2, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol conv5_1_w("conv5_1_weight"), conv5_1_b("conv5_1_bias");
  Symbol conv5_1 = Convolution("conv5_1", pool4, conv5_1_w, conv5_1_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu5_1 = Activation("relu5_1", conv5_1, "relu");

  Symbol conv5_2_w("conv5_2_weight"), conv5_2_b("conv5_2_bias");
  Symbol conv5_2 = Convolution("conv5_2", relu5_1, conv5_2_w, conv5_2_b,
                               Shape(3, 3), 512, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu5_2 = Activation("relu5_2", conv5_2, "relu");
  Symbol pool5 = Pooling("pool5", relu5_2, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2, 2));

  Symbol flatten = Flatten("flatten", pool5);
  Symbol fc6_w("fc6_weight"), fc6_b("fc6_bias");
  Symbol fc6 = FullyConnected("fc6", flatten, fc6_w, fc6_b, 4096);
  Symbol relu6 = Activation("relu6", fc6, "relu");
  Symbol drop6 = Dropout("drop6", relu6, 0.5);

  Symbol fc7_w("fc7_weight"), fc7_b("fc7_bias");
  Symbol fc7 = FullyConnected("fc7", drop6, fc7_w, fc7_b, 4096);
  Symbol relu7 = Activation("relu7", fc7, "relu");
  Symbol drop7 = Dropout("drop7", relu7, 0.5);

  Symbol fc8_w("fc8_weight"), fc8_b("fc8_bias");
  Symbol fc8 = FullyConnected("fc8", drop7, fc8_w, fc8_b, num_classes);
  return SoftmaxOutput("softmax", fc8, data_label);
}

Symbol LenetSymbol(int num_classes) {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");
  Symbol conv1_w("convolution0_weight"), conv1_b("convolution0_bias");
  Symbol conv2_w("convolution1_weight"), conv2_b("convolution1_bias");

  Symbol fc1_w("fullyconnected0_weight"), fc1_b("fullyconnected0_bias");
  Symbol fc2_w("fullyconnected1_weight"), fc2_b("fullyconnected1_bias");

  Symbol conv1 = Convolution("convolution0", data, conv1_w, conv1_b, Shape(5, 5), 20);
  Symbol tanh1 = Activation("activation0", conv1, "tanh");
  Symbol pool1 = Pooling("pooling0", tanh1, Shape(2, 2), PoolingPoolType::max, false, Shape(2, 2));

  Symbol conv2 = Convolution("convolution1", pool1, conv2_w, conv2_b, Shape(5, 5), 50);
  Symbol tanh2 = Activation("activation1", conv2, "tanh");
  Symbol pool2 = Pooling("pooling1", tanh2, Shape(2, 2), PoolingPoolType::max, false, Shape(2, 2));

  Symbol flatten = Flatten("flatten0", pool2);
  Symbol fc1 = FullyConnected("fullyconnected0", flatten, fc1_w, fc1_b, 500);
  Symbol tanh3 = Activation("activation2", fc1, "tanh");
  Symbol fc2 = FullyConnected("fullyconnected1", tanh3, fc2_w, fc2_b, num_classes);

  Symbol lenet = SoftmaxOutput("softmax", fc2, data_label);

  return lenet;
}

Symbol getConv(const std::string & name, Symbol data,
               int  num_filter,
               Shape kernel, Shape stride, Shape pad,
               bool with_relu,
               mx_float bn_momentum) {
  Symbol conv_w(name + "_weight");
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
    Symbol shortcut_w(name + "_proj_weight");
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
  Symbol data_label = Symbol::Variable("softmax_label");

  Symbol zscore = BatchNorm("zscore", data, 0.001, bn_momentum);

  Symbol conv = getConv("conv0", zscore, num_filter,
                        Shape(3, 3), Shape(1, 1), Shape(1, 1),
                        true, bn_momentum);

  Symbol body = getBody(conv, num_level, num_block, num_filter, bn_momentum);

  Symbol pool = Pooling("pooling0", body, pool_kernel, PoolingPoolType::avg);

  Symbol flat = Flatten("flatten0", pool);

  Symbol fc_w("fc_weight"), fc_b("fc_bias");
  Symbol fc = FullyConnected("fc", flat, fc_w, fc_b, num_class);

  return SoftmaxOutput("softmax", fc, data_label);
}

Symbol Inception7A(Symbol data, int num_1x1, int num_3x3_red, int num_3x3_1,
                   int num_3x3_2, int num_5x5_red, int num_5x5,
                   PoolingPoolType pool, int proj,
                   const std::string & name) {
  Symbol tower_1x1 = ConvFactoryNoBias(data, num_1x1, Shape(1, 1),
                                       Shape(1, 1),
                                       Shape(0, 0),
                                       name + "_conv");
  Symbol tower_5x5 = ConvFactoryNoBias(data,
                                       num_5x5_red, Shape(1, 1),
                                       Shape(1, 1), Shape(0, 0),
                                       name + "_tower", "_conv");
  tower_5x5 = ConvFactoryNoBias(tower_5x5, num_5x5, Shape(5, 5),
                                Shape(1, 1), Shape(2, 2),
                                name + "_tower", "_conv_1");
  Symbol tower_3x3 = ConvFactoryNoBias(data,
                                       num_3x3_red, Shape(1, 1),
                                       Shape(1, 1), Shape(0, 0),
                                       name + "_tower_1", "_conv");
  tower_3x3 = ConvFactoryNoBias(tower_3x3,
                                num_3x3_1, Shape(3, 3),
                                Shape(1, 1), Shape(1, 1),
                                name + "_tower_1" , "_conv_1");
  tower_3x3 = ConvFactoryNoBias(tower_3x3,
                                num_3x3_2, Shape(3, 3),
                                Shape(1, 1), Shape(1, 1),
                                name + "_tower_1", "_conv_2");
  Symbol pooling =
      Pooling(PoolingPoolTypeValues[static_cast<int>(pool)] + "_pool_" +name + "_pool",
              data, Shape(3, 3), pool,
              false, Shape(1, 1), Shape(1, 1));
  Symbol cproj = ConvFactoryNoBias(pooling,
                                   proj, Shape(1, 1),
                                   Shape(1, 1), Shape(0, 0),
                                   name + "_tower_2", "_conv");
  std::vector<Symbol> concat_lst;
  concat_lst.push_back(tower_1x1);
  concat_lst.push_back(tower_5x5);
  concat_lst.push_back(tower_3x3);
  concat_lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", concat_lst, concat_lst.size());
}

Symbol Inception7B(Symbol data, int num_3x3, int num_d3x3_red,
                   int num_d3x3_1, int num_d3x3_2,
                   PoolingPoolType pool, const std::string & name) {
  Symbol tower_3x3 = ConvFactoryNoBias(data, num_3x3,
                                       Shape(3, 3), Shape(2, 2), Shape(0, 0),
                                       name + "_conv");
  Symbol tower_d3x3 = ConvFactoryNoBias(data, num_d3x3_red,
                                        Shape(1, 1), Shape(1, 1), Shape(0, 0),
                                        name + "_tower", "_conv");
  tower_d3x3 = ConvFactoryNoBias(tower_d3x3, num_d3x3_1,
                                 Shape(3, 3), Shape(1, 1), Shape(1, 1),
                                 name + "_tower", "_conv_1");
  tower_d3x3 = ConvFactoryNoBias(tower_d3x3, num_d3x3_2,
                                 Shape(3, 3), Shape(2, 2), Shape(0, 0),
                                 name + "_tower", "_conv_2");
  Symbol pooling = Pooling("max_pool_" + name + "_pool",
                           data, Shape(3, 3),
                           PoolingPoolType::max, false,
                           Shape(2, 2), Shape(0, 0));
  std::vector<Symbol> lst;
  lst.push_back(tower_3x3);
  lst.push_back(tower_d3x3);
  lst.push_back(pooling);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol Inception7C(Symbol data, int num_1x1, int num_d7_red, int num_d7_1,
                   int num_d7_2, int num_q7_red, int num_q7_1,
                   int num_q7_2, int num_q7_3, int num_q7_4,
                   PoolingPoolType pool,
                   int proj, const std::string & name) {
  Symbol tower_1x1 = ConvFactoryNoBias(data, num_1x1, Shape(1, 1),
                                       Shape(1, 1), Shape(0, 0),
                                       name + "_conv");
  Symbol tower_d7 = ConvFactoryNoBias(data, num_d7_red, Shape(1, 1),
                                      Shape(1, 1), Shape(0, 0),
                                      name + "_tower", "_conv");
  tower_d7 = ConvFactoryNoBias(tower_d7, num_d7_1, Shape(1, 7),
                               Shape(1, 1), Shape(0, 3),
                               name + "_tower", "_conv_1");
  tower_d7 = ConvFactoryNoBias(tower_d7, num_d7_2, Shape(7, 1),
                               Shape(1, 1), Shape(3, 0),
                               name + "_tower", "_conv_2");
  Symbol tower_q7 = ConvFactoryNoBias(data, num_q7_red, Shape(1, 1),
                                      Shape(1, 1), Shape(0, 0),
                                      name + "_tower_1", "_conv");
  tower_q7 = ConvFactoryNoBias(tower_q7, num_q7_1, Shape(7, 1),
                               Shape(1, 1), Shape(3, 0),
                               name + "_tower_1", "_conv_1");
  tower_q7 = ConvFactoryNoBias(tower_q7, num_q7_2, Shape(1, 7),
                               Shape(1, 1), Shape(0, 3),
                               name + "_tower_1", "_conv_2");
  tower_q7 = ConvFactoryNoBias(tower_q7, num_q7_3, Shape(7, 1),
                               Shape(1, 1), Shape(3, 0),
                               name + "_tower_1", "_conv_3");
  tower_q7 = ConvFactoryNoBias(tower_q7, num_q7_4, Shape(1, 7),
                               Shape(1, 1), Shape(0, 3),
                               name + "_tower_1", "_conv_4");
  Symbol pooling =
      Pooling(PoolingPoolTypeValues[static_cast<int>(pool)] + "_pool_" +name + "_pool",
              data, Shape(3, 3), pool, false,
              Shape(1, 1), Shape(1, 1));
  Symbol cproj = ConvFactoryNoBias(pooling, proj, Shape(1, 1),
                                   Shape(1, 1), Shape(0, 0),
                                   name + "_tower_2", "_conv");
  // concat
  std::vector<Symbol> lst;
  lst.push_back(tower_1x1);
  lst.push_back(tower_d7);
  lst.push_back(tower_q7);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol Inception7D(Symbol data,
                   int num_3x3_red, int num_3x3, int num_d7_3x3_red,
                   int num_d7_1, int num_d7_2, int num_d7_3x3,
                   PoolingPoolType pool,
                   const std::string & name) {
  Symbol tower_3x3 = ConvFactoryNoBias(data, num_3x3_red, Shape(1, 1),
                                       Shape(1, 1), Shape(0, 0),
                                       name + "_tower", "_conv");
  tower_3x3 = ConvFactoryNoBias(tower_3x3, num_3x3, Shape(3, 3),
                                Shape(2, 2), Shape(0, 0),
                                name + "_tower", "_conv_1");
  Symbol tower_d7_3x3 = ConvFactoryNoBias(data, num_d7_3x3_red, Shape(1, 1),
                                          Shape(1, 1), Shape(0, 0),
                                          name + "_tower_1", "_conv");
  tower_d7_3x3 = ConvFactoryNoBias(tower_d7_3x3, num_d7_1, Shape(1, 7),
                                   Shape(1, 1), Shape(0, 3),
                                   name + "_tower_1", "_conv_1");
  tower_d7_3x3 = ConvFactoryNoBias(tower_d7_3x3, num_d7_2, Shape(7, 1),
                                   Shape(1, 1), Shape(3, 0),
                                   name + "_tower_1", "_conv_2");
  tower_d7_3x3 = ConvFactoryNoBias(tower_d7_3x3, num_d7_3x3, Shape(3, 3),
                                   Shape(2, 2), Shape(0, 0),
                                   name + "_tower_1", "_conv_3");
  Symbol pooling =
      Pooling(PoolingPoolTypeValues[static_cast<int>(pool)] + "_pool_" +name + "_pool",
              data, Shape(3, 3), pool, false, Shape(2, 2));
  // concat
  std::vector<Symbol> lst;
  lst.push_back(tower_3x3);
  lst.push_back(tower_d7_3x3);
  lst.push_back(pooling);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol Inception7E(Symbol data,
                   int num_1x1, int num_d3_red, int num_d3_1,
                   int num_d3_2, int num_3x3_d3_red, int num_3x3,
                   int num_3x3_d3_1, int num_3x3_d3_2,
                   PoolingPoolType pool,
                   int proj, const std::string & name) {
  Symbol tower_1x1 = ConvFactoryNoBias(data, num_1x1, Shape(1, 1),
                                       Shape(1, 1), Shape(0, 0),
                                       name + "_conv");
  Symbol tower_d3 = ConvFactoryNoBias(data, num_d3_red, Shape(1, 1),
                                      Shape(1, 1), Shape(0, 0),
                                      name + "_tower", "_conv");
  Symbol tower_d3_a = ConvFactoryNoBias(tower_d3, num_d3_1, Shape(1, 3),
                                        Shape(1, 1), Shape(0, 1),
                                        name + "_tower", "_mixed_conv");
  Symbol tower_d3_b = ConvFactoryNoBias(tower_d3, num_d3_2, Shape(3, 1),
                                        Shape(1, 1), Shape(1, 0),
                                        name + "_tower", "_mixed_conv_1");
  Symbol tower_3x3_d3 = ConvFactoryNoBias(data, num_3x3_d3_red, Shape(1, 1),
                                          Shape(1, 1), Shape(0, 0),
                                          name + "_tower_1", "_conv");
  tower_3x3_d3 = ConvFactoryNoBias(tower_3x3_d3, num_3x3, Shape(3, 3),
                                   Shape(1, 1), Shape(1, 1),
                                   name + "_tower_1", "_conv_1");
  Symbol tower_3x3_d3_a = ConvFactoryNoBias(tower_3x3_d3, num_3x3_d3_1, Shape(1, 3),
                                            Shape(1, 1), Shape(0, 1),
                                            name + "_tower_1", "_mixed_conv");
  Symbol tower_3x3_d3_b = ConvFactoryNoBias(tower_3x3_d3, num_3x3_d3_2, Shape(3, 1),
                                            Shape(1, 1), Shape(1, 0),
                                            name + "_tower_1", "_mixed_conv_1");
  Symbol pooling =
      Pooling(PoolingPoolTypeValues[static_cast<int>(pool)] + "_pool_" +name + "_pool", data,
              Shape(3, 3), pool, false,
              Shape(1, 1), Shape(1, 1));
  Symbol cproj = ConvFactoryNoBias(pooling, proj, Shape(1, 1),
                                   Shape(1, 1), Shape(0, 0),
                                   name + "_tower_2", "_conv");
  // concat
  std::vector<Symbol> lst;
  lst.push_back(tower_1x1);
  lst.push_back(tower_d3_a);
  lst.push_back(tower_d3_b);
  lst.push_back(tower_3x3_d3_a);
  lst.push_back(tower_3x3_d3_b);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

mxnet::cpp::Symbol InceptionV3Symbol(int num_classes) {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");
  // stage 1
  Symbol conv = ConvFactoryNoBias(data, 32, Shape(3, 3), Shape(2, 2),
                                  Shape(0, 0), "conv");
  Symbol conv_1 = ConvFactoryNoBias(conv, 32, Shape(3, 3),
                                    Shape(1, 1), Shape(0, 0),
                                    "conv_1");
  Symbol conv_2 = ConvFactoryNoBias(conv_1, 64, Shape(3, 3), Shape(1, 1),
                                    Shape(1, 1), "conv_2");
  Symbol pool = Pooling("pool", conv_2, Shape(3, 3),
                        PoolingPoolType::max,
                        false, Shape(2, 2), Shape(0, 0));
  // stage 2
  Symbol conv_3 = ConvFactoryNoBias(pool, 80, Shape(1, 1),
                                    Shape(1, 1), Shape(0, 0),
                                    "conv_3");
  Symbol conv_4 = ConvFactoryNoBias(conv_3, 192, Shape(3, 3),
                                    Shape(1, 1), Shape(0, 0),
                                    "conv_4");
  Symbol pool1 = Pooling("pool1", conv_4,
                         Shape(3, 3), PoolingPoolType::max, false,
                         Shape(2, 2));
  // stage 3
  Symbol in3a = Inception7A(pool1, 64, 64, 96, 96, 48, 64,
                            PoolingPoolType::avg, 32, "mixed");
  Symbol in3b = Inception7A(in3a, 64, 64, 96, 96, 48, 64,
                            PoolingPoolType::avg, 64, "mixed_1");
  Symbol in3c = Inception7A(in3b, 64, 64, 96, 96, 48, 64,
                            PoolingPoolType::avg, 64, "mixed_2");
  Symbol in3d = Inception7B(in3c, 384, 64, 96, 96,
                            PoolingPoolType::max, "mixed_3");
  // stage 4
  Symbol in4a = Inception7C(in3d, 192, 128, 128, 192, 128, 128, 128, 128, 192,
                            PoolingPoolType::avg, 192, "mixed_4");
  Symbol in4b = Inception7C(in4a, 192, 160, 160, 192, 160, 160, 160, 160, 192,
                            PoolingPoolType::avg, 192, "mixed_5");
  Symbol in4c = Inception7C(in4b, 192, 160, 160, 192, 160, 160, 160, 160, 192,
                            PoolingPoolType::avg, 192, "mixed_6");
  Symbol in4d = Inception7C(in4c, 192, 192, 192, 192, 192, 192, 192, 192, 192,
                            PoolingPoolType::avg, 192, "mixed_7");
  Symbol in4e = Inception7D(in4d, 192, 320, 192, 192, 192, 192,
                            PoolingPoolType::max, "mixed_8");
  // stage 5
  Symbol in5a = Inception7E(in4e, 320, 384, 384, 384, 448, 384, 384, 384,
                            PoolingPoolType::avg, 192, "mixed_9");
  Symbol in5b = Inception7E(in5a, 320, 384, 384, 384, 448, 384, 384, 384,
                            PoolingPoolType::max, 192, "mixed_10");
  // pool
  pool = Pooling("global_pool", in5b, Shape(8, 8),
                 PoolingPoolType::avg, false);
  Symbol flatten = Flatten("flatten", pool);
  Symbol fc1 = FullyConnected("fc1", flatten, num_classes);
  return SoftmaxOutput("softmax", fc1, data_label);
}

Symbol ConvModule(const std::string & name,
                  Symbol net, Shape kernel_size,
                  Shape pad_size, int filter_count,
                  Shape stride, int work_space,
                  bool batch_norm, bool down_pool,
                  bool up_pool, const std::string & act_type,
                  bool convolution) {
  if (up_pool) {
    net = Operator("Deconvolution")
        .SetParam("kernel", Shape(2, 2))
        .SetParam("num_filter", filter_count)
        .SetParam("stride", Shape(2, 2))
        .SetParam("pad", Shape(0, 0))
        .SetParam("workspace", work_space)
        .SetInput("data", net)
        .CreateSymbol(name + "_deconv");
    net = BatchNorm(name + "_bn", net);
    if (act_type != "") {
      net = Activation(name + "_act", net, act_type);
    }
  }

  if (convolution) {
    net = Operator("Convolution")
        .SetParam("kernel", kernel_size)
        .SetParam("num_filter", filter_count)
        .SetParam("stride", stride)
        .SetParam("pad", pad_size)
        .SetParam("workspace", work_space)
        .SetInput("data", net)
        .CreateSymbol(name + "_conv");
  }

  if (batch_norm) {
    net = BatchNorm(name + "_bn", net);
  }

  if (act_type != "") {
    net = Activation(name + "_act", net, act_type);
  }

  if (down_pool) {
    net = Operator("Pooling")
        .SetParam("kernel", Shape(2, 2))
        .SetParam("pool_type", "max")
        .SetParam("stride", Shape(2, 2))
        .SetInput("data", net)
        .CreateSymbol(name + "_max_pool");
  }

  return net;
}

Symbol UNetSymbol() {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");
  Shape kernel_size = Shape(3, 3);
  Shape pad_size = Shape(1, 1);
  int filter_count = 32;
  Symbol pool1 = ConvModule("pool1", data, kernel_size, pad_size,
                            filter_count, Shape(1, 1),
                            2048, true, true, false,
                            "relu", true);
  Symbol net = pool1;
  Symbol pool2 = ConvModule("pool2", net, kernel_size, pad_size,
                            filter_count * 2, Shape(1, 1),
                            2048, true, true);
  net = pool2;
  Symbol pool3 = ConvModule("pool3", net, kernel_size, pad_size,
                            filter_count * 4, Shape(1, 1),
                            2048, true, true);
  net = pool3;
  Symbol pool4 = ConvModule("pool4", net, kernel_size, pad_size,
                            filter_count * 4, Shape(1, 1),
                            2048, true, true);
  net = pool4;
  net = Dropout("pool4_drop", net);
  Symbol pool5 = ConvModule("pool5", net, kernel_size, pad_size,
                            filter_count * 8, Shape(1, 1),
                            2048, true, true);
  net = pool5;
  net = ConvModule("pool5_conv1", net, kernel_size, pad_size,
                   filter_count * 4, Shape(1, 1),
                   2048, true, false, true);
  net = ConvModule("pool5_conv2", net, kernel_size, pad_size,
                   filter_count * 4, Shape(1, 1),
                   2048, true, false, true);
  net = ConvModule("pool5_conv3", net, Shape(4, 4), Shape(0, 0), filter_count * 4);
  std::vector<Symbol> lst;
  lst.push_back(pool3);
  lst.push_back(net);
  net = Concat("pool3_concat", lst, lst.size());
  net = Dropout("pool3_drop", net);
  net = ConvModule("pool3_conv1", net, kernel_size, pad_size, filter_count * 4);
  net = ConvModule("pool3_conv2", net, kernel_size, pad_size, filter_count * 4,
                   Shape(1, 1), 2048, true, false, true);

  lst.clear();
  lst.push_back(pool2);
  lst.push_back(net);
  net = Concat("pool2_concat", lst, lst.size());
  net = Dropout("pool2_drop", net);
  net = ConvModule("pool2_conv1", net, kernel_size, pad_size, filter_count * 4);
  net = ConvModule("pool2_conv2", net, kernel_size, pad_size, filter_count * 4,
                   Shape(1, 1), 2048, true, false, true);
  lst.clear();
  lst.push_back(pool1);
  lst.push_back(net);
  net = Concat("pool1_concat", lst, lst.size());
  net = Dropout("pool1_drop", net);
  net = ConvModule("pool1_conv1", net, kernel_size, pad_size, filter_count * 2);
  net = ConvModule("pool1_conv2", net, kernel_size, pad_size, filter_count * 2,
                   Shape(1, 1), 2048, true, false, true);
  net = ConvModule("pool1_conv3", net, kernel_size, pad_size, 1, Shape(1, 1),
                   2048, false, false, false, "");
  net = Flatten("flatten", net);
  return LogisticRegressionOutput("softmax", net, data_label);
}

Symbol MLPSymbol(const std::vector<int> &layerSize,
                 const std::vector<std::string> &activations,
                 int num_classes,
                 double input_dropout,
                 const std::vector<double> &hidden_dropout)
{
  Symbol act = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");
  Symbol inputdropout = Dropout("dropout0", act, input_dropout);
  std::vector<Symbol> fc_w, fc_b, fc, drop;
  int nLayers = layerSize.size();

  for (int i = 0; i < nLayers; i++) {
    fc_w.push_back(Symbol("fc" + std::to_string(i + 1) + "_w"));
    fc_b.push_back(Symbol("fc" + std::to_string(i + 1) + "_b"));
    fc.push_back(FullyConnected("fc" + std::to_string(i + 1), act, fc_w[i], fc_b[i], layerSize[i]));
    act = Activation(activations[i] + std::to_string(i + 1), fc[i], activations[i].c_str());
    drop.push_back(Dropout("dropout" + std::to_string(i + 1), fc[i], hidden_dropout[i]));
  }
  fc_w.push_back(Symbol("fc" + std::to_string(nLayers + 1) + "_w"));
  fc_b.push_back(Symbol("fc" + std::to_string(nLayers + 1) + "_b"));
  fc.push_back(FullyConnected("fc" + std::to_string(nLayers + 1),
                              act, fc_w[nLayers], fc_b[nLayers], num_classes));

  return SoftmaxOutput("softmax", fc[nLayers], data_label);
}
