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

class MLPNative {
 public:
  MLPNative();
  void setLayers(int * lsize, int nsize, int n);
  void setAct(char ** act);
  void setData(float * data, int dimX1, int dimX2);
  void setLabel(float *, int);
  void setLR(float lr) {learning_rate = lr;}
  void setWD(float wd) {weight_decay = wd;}
  void setBatch(int batch) {batch_size = batch;}

  void build_mlp();
  void train();
  void train(float learning_rate, float weight_decay);
  float compAccuracy();
  std::vector<float> pred();
  std::vector<float> pred(float *, int);

 private:
  int nLayers, nOut;
  int dimX1, dimX2, dimY;
  mx_float learning_rate = 0.001;
  mx_float weight_decay = 0.001;
  int batch_size = 15;
  std::vector<int> layerSize;
  std::vector<std::string> activations;
  std::vector<mx_float> label;
  std::map<std::string, mxnet::cpp::NDArray> args_map;
  mxnet::cpp::Symbol sym_network;
  mxnet::cpp::NDArray array_x;
  mxnet::cpp::NDArray array_y;
  mxnet::cpp::Context ctx_dev;
};

#endif
