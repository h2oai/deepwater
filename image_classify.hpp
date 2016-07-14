/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __H2O_IMAGE_CLASSIFY_H__
#define __H2O_IMAGE_CLASSIFY_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "network_def.hpp"

class ImageClassify{
 public:
  ImageClassify();
  ~ImageClassify();
  void buildNet(int num_classes, int batch_size);
  void loadModel(char * model_path);
  void setLR(float lr) {learning_rate = lr;}
  void setWD(float wd) {weight_decay = wd;}
  std::vector<float> train(float * data, float * label, bool is_train);

 private:
  int width, height, batch_size, num_classes;
  float learning_rate, weight_decay;

  std::map<std::string, mxnet::cpp::NDArray> args_map;
  mxnet::cpp::Symbol inception_bn_net;
  mxnet::cpp::Executor * exec;
  mxnet::cpp::Optimizer * opt;
};

#endif
