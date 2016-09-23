/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __H2O_NUMERIC_TRAIN_H__
#define __H2O_NUMERIC_TRAIN_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "network_def.hpp"

class NumericTrain{
 public:
  NumericTrain(int ncols, int device = 0, int seed = 0, bool gpu = true);
  void buildNet(int num_classes, int batch_size, char * net_name,
		 int num_hidden,
                 int *hidden,
                 char**activations,
                 float input_dropout,
                 float *hidden_dropout);
  void setOptimizer(int num_classes, int batch_size);
  void setSeed(int seed);
  void loadModel(char * model_path);
  void saveModel(char * model_path);
  void loadParam(char * param_path);
  void saveParam(char * param_path);
  const char * toJson();

  void setLR(float lr) {learning_rate = lr;}
  void setWD(float wd) {weight_decay = wd;}
  void setMomentum(float m) {
    momentum = m;
    if (opt.get() != nullptr) opt->SetParam("momentum", momentum);
  }
  void setClipGradient(float c) {
    clip_gradient = c;
    if (opt.get() != nullptr) opt->SetParam("clip_gradient", clip_gradient);
  }

  std::vector<float> train(float * data, float * label);
  std::vector<float> predict(float * data, float * label);
  std::vector<float> predict(float * data);

 private:
  int num_cols, batch_size, num_classes;
  float learning_rate, weight_decay, momentum, clip_gradient;

  std::map<std::string, mxnet::cpp::NDArray> args_map;
  std::map<std::string, mxnet::cpp::NDArray> aux_map;
  mxnet::cpp::Symbol mxnet_sym;
  std::unique_ptr<mxnet::cpp::Executor> exec;
  std::unique_ptr<mxnet::cpp::Optimizer> opt;
  bool is_built;
  mxnet::cpp::Context ctx_dev;

  std::vector<float> execute(float * data, float * label, bool is_train);

  // prediction probs
  std::vector<float> preds;
};

#endif
