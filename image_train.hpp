/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __H2O_IMAGE_TRAIN_H__
#define __H2O_IMAGE_TRAIN_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "network_def.hpp"

class ImageTrain{
 public:
  ImageTrain();
  ~ImageTrain();
  void buildNet(int num_classes, int batch_size, char * net_name);
  void loadModel(char * model_path);
  void loadParam(char * param_path);
  void saveParam(char * param_path);
  void setLR(float lr) {learning_rate = lr;}
  void setWD(float wd) {weight_decay = wd;}
  std::vector<float> train(float * data, float * label);
  std::vector<float> predict(float * data, float * label);

 private:
  int width, height, batch_size, num_classes;
  float learning_rate, weight_decay;

  std::map<std::string, mxnet::cpp::NDArray> args_map;
  mxnet::cpp::Symbol mxnet_sym;
  mxnet::cpp::Executor * exec;
  mxnet::cpp::Optimizer * opt;
  bool is_built;
  mxnet::cpp::Context ctx_dev;

  std::vector<float> execute(float * data, float * label, bool is_train);
};

#endif
