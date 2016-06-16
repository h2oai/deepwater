#ifndef __H2O_MPL_H__
#define __H2O_MPL_H__

#include <memory>
#include "MxNetCpp.h"

class MLPClass {
 public:
  MLPClass();
  void setLayers(int * lsize, int nsize);
  void setAct(char **);
  void setX(float *, int *, int);
  void setLabel(float *, int);

  void buildnn();
  float train(bool);

 private:
  int nLayers, dimX1, dimX2, dimY;
  std::vector<int> layerSize;
  std::vector<std::string> activations;
  std::vector<mxnet::cpp::Symbol> weights;
  std::vector<mxnet::cpp::Symbol> biases;
  std::vector<mxnet::cpp::Symbol> outputs;
  std::shared_ptr<mxnet::cpp::Executor> exe;
  mxnet::cpp::Symbol sym_x, sym_label, sym_out;
  mxnet::cpp::NDArray array_x;
  mxnet::cpp::NDArray array_y;
  mxnet::cpp::Context ctx_dev;
};

#endif  
