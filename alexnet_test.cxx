
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "MxNetCpp.h"

using namespace mxnet::cpp;

Symbol AlexnetSymbol(int num_classes) {

  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
  Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b,
                             Shape(11, 11), 96, Shape(4, 4));
  Symbol relu1 = Activation("relu1", conv1, "relu");
  Symbol pool1 = Pooling("pool1", relu1, Shape(3, 3),
                         PoolingPoolType::max, false, Shape(2,2));
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

int main(int argc, char const *argv[]) {
  return 0;
}
