#ifndef __H2O_MPL_H__
#define __H2O_MPL_H__

#include <memory>
#include "include/MxNetCpp.h"

class MLPNative {
 public:
  MLPNative();
  void setLayers(int * lsize, int nsize, int n);
  void setAct(char **);
  void setData(mx_float *, int, int);
  void setLabel(mx_float *, int);
  void setLR(mx_float lr) {learning_rate = lr;}
  void setWD(mx_float wd) {weight_decay = wd;}
  void setBatch(int batch) {batch_size = batch;}

  void build_mlp();
  mx_float* train();

 private:
  int nLayers, nOut;
  int dimX1, dimX2, dimY;
  mx_float learning_rate = 1e-4;
  mx_float weight_decay = 1e-4;
  int batch_size = 20;
  std::vector<int> layerSize;
  std::vector<std::string> activations;
  std::vector<mx_float> label;
  mx_float * pred;
  //  std::shared_ptr<mxnet::cpp::Executor> exe;
  std::map<std::string, mxnet::cpp::NDArray> args_map;
  //mxnet::cpp::Optimizer opt;
  mxnet::cpp::Symbol sym_network;
  mxnet::cpp::NDArray array_x;
  mxnet::cpp::NDArray array_y;
  mxnet::cpp::Context ctx_dev;
};

#endif  
