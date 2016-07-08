#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "MxNetCpp.h"

using namespace mxnet::cpp;

Symbol VGGSymbol(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol conv1_1_w("conv1_1_w"), conv1_1_b("conv1_1_b");
  Symbol conv1_1 = Convolution("conv1_1", data, conv1_1_w, conv1_1_b,
                               Shape(3, 3), 64, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu1_1 = Activation("relu1_1", conv1_1, "relu");
  Symbol pool1 = Pooling("pool1", relu1_1, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2,2));

  Symbol conv2_1_w("conv2_1_w"), conv2_1_b("conv2_1_b");
  Symbol conv2_1 = Convolution("conv2_1", pool1, conv2_1_w, conv2_1_b,
                               Shape(3, 3), 128, Shape(1, 1),
                               Shape(1, 1), Shape(1, 1));
  Symbol relu2_1 = Activation("relu2_1", conv2_1, "relu");
  Symbol pool2 = Pooling("pool2", relu2_1, Shape(2, 2), PoolingPoolType::max,
                         false, Shape(2,2));

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
                         false, Shape(2,2));

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
                         false, Shape(2,2));

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
                         false, Shape(2,2));

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

int main(int argc, char const *argv[]) {


}
