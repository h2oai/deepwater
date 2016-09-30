/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <map>
#include <iostream>

#include "mxnet/io.h"
#include "mxnet/ndarray.h"
#include "mxnet/base.h"
#include "mxnet/operator.h"
#include "mxnet/operator_util.h"
#include "mxnet/symbolic.h"
#include "mxnet/optimizer.h"

#include "util.hpp"

int main() {
  std::cout << "#OperatorPropertyReg" << std::endl;
  std::vector<const mxnet::OperatorPropertyReg * > vec =
      dmlc::Registry<mxnet::OperatorPropertyReg>::List();

  for (size_t i = 0; i < vec.size(); i++) {
    FunctionRegInfo<mxnet::OperatorPropertyReg>(vec[i]);
  }

  std::cout << "#OptimizerReg" << std::endl;
  std::vector<const mxnet::OptimizerReg * > vec2 =
      dmlc::Registry<mxnet::OptimizerReg>::List();

  for (size_t i = 0; i < vec2.size(); i++) {
    FunctionRegInfo<mxnet::OptimizerReg>(vec2[i]);
  }

  std::cout << "#DataIteratorReg" << std::endl;
  std::vector<const mxnet::DataIteratorReg * > vec3 =
      dmlc::Registry<mxnet::DataIteratorReg>::List();

  for (size_t i = 0; i < vec3.size(); i++) {
    FunctionRegInfo<mxnet::DataIteratorReg>(vec3[i]);
  }
}
