/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __H2O_MPL_H__
#define __H2O_MPL_H__

#include <memory>
#include <map>
#include <string>
#include <vector>
#include "include/MxNetCpp.h"

class MLP {
 public:
  MLP();
  ~MLP();
  void setLayers(int * lsize, int nsize, int num_classes);
  void setAct(char ** act);
  void setLR(float lr) {learning_rate = lr;}
  void setWD(float wd) {weight_decay = wd;}
  void setBatch(int b) {batch_size = b;}
  void setDimX(int x) {dimX = x;}

  void buildMLP();
  void setSeed(int seed);
  std::vector<float> train(float * data, float * label);
  std::vector<float> predict(float * data, float * label);
  std::vector<float> predict(float * data);

  void saveParam(char * param_path);
  void loadParam(char * param_path);
  void saveModel(char * model_path);
  void loadModel(char * model_path);

 private:
  int nLayers, num_classes, batch_size, dimX;
  mx_float learning_rate = 1e-3;
  mx_float weight_decay = 1e-3;
  std::vector<int> layerSize;
  std::vector<std::string> activations;

  std::map<std::string, mxnet::cpp::NDArray> args_map;
  mxnet::cpp::Symbol sym_network;
  mxnet::cpp::Executor * exec;
  mxnet::cpp::Optimizer * opt;
  mxnet::cpp::Context ctx_dev;
  bool is_built;

  std::vector<float> execute(float * data, float * label, bool is_train);

  // prediction probs
  std::vector<float> preds;
};

#endif
