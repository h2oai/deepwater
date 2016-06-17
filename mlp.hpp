#ifndef __H2O_MPL_H__
#define __H2O_MPL_H__

#include <memory>
#include "include/MxNetCpp.h"

class MLPNative {
 public:
  MLPNative();
  void setLayers(int * lsize, int nsize);
  void setData(mx_float *, int *, int);
  void setLabel(mx_float *, int);

  void build_mlp();
  mx_float* train();

 private:
  int nLayers;
  int dimX1, dimX2, dimY;
  mx_float learning_rate = 1e-4;
  mx_float weight_decay = 1e-4;
  std::vector<int> layerSize;
  std::vector<mx_float> label;
  mx_float * pred;
  std::shared_ptr<mxnet::cpp::Executor> exe;
  mxnet::cpp::Symbol sym_network;
  mxnet::cpp::NDArray array_x;
  mxnet::cpp::NDArray array_y;
  mxnet::cpp::Context ctx_dev;
};

#endif  
