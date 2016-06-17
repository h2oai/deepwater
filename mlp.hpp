#ifndef __H2O_MPL_H__
#define __H2O_MPL_H__

#include <memory>
#include "include/MxNetCpp.h"

class MLPClass {
 public:
  MLPClass();
  void setLayers(int * lsize, int nsize);
  void setX(float *, int *, int);
  void setLabel(float *, int);

  void buildnn();
  float train(int iter, bool verbose);

 private:
  int nLayers;
  int dimX1, dimX2;
  int  dimY;
  mx_float learning_rate = 0.0001;
  std::vector<int> layerSize;
  std::vector<float> label;
  float * pred;
  std::vector<mxnet::cpp::Symbol> weights;
  std::vector<mxnet::cpp::Symbol> biases;
  std::vector<mxnet::cpp::Symbol> outputs;
  std::shared_ptr<mxnet::cpp::Executor> exe;
  mxnet::cpp::Symbol sym_x, sym_label, sym_out;
  mxnet::cpp::NDArray array_x;
  mxnet::cpp::NDArray array_y;
  mxnet::cpp::Context ctx_dev;
  std::vector<mxnet::cpp::NDArray> in_args;
  std::vector<mxnet::cpp::NDArray> arg_grad_store;
  std::vector<mxnet::cpp::OpReqType> grad_req_type;
  std::vector<mxnet::cpp::NDArray> aux_states;
};

#endif  
