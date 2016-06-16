/*!
*  Copyright (c) 2016 by Contributors
* \file optimizer.hpp
* \brief implementation of optimizer
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNETCPP_OPTIMIZER_HPP
#define MXNETCPP_OPTIMIZER_HPP

#include <numeric>
#include <map>
#include <string>
#include <vector>
#include "optimizer.h"

namespace mxnet {
namespace cpp {

Optimizer::Optimizer(const std::string &opt_type, mx_float learning_rate, mx_float weight_decay)
  :init_(false), learning_rate_(learning_rate), weight_decay_(weight_decay), opt_type_(opt_type) {
  MXOptimizerFindCreator(opt_type.c_str(), &creator_);
}

void Optimizer::Update(int index, NDArray weight, NDArray grad) {
  if (!init_) {
    std::vector<const char *> param_keys;
    std::vector<const char *> param_values;
    for (const auto &k_v : params_) {
      param_keys.push_back(k_v.first.c_str());
      param_values.push_back(k_v.second.c_str());
    }
    MXOptimizerCreateOptimizer(creator_, params_.size(), param_keys.data(),
                               param_values.data(), &handle_);
    init_ = true;
  }
  MXOptimizerUpdate(handle_, index, weight.GetHandle(), grad.GetHandle(),
      learning_rate_, weight_decay_);
}

std::string Optimizer::Serialize() const {
  using ValueType = std::map<std::string, std::string>::value_type;
  auto params = params_;
  params.emplace("opt_type", opt_type_);
  params.emplace("learning_rate", std::to_string(learning_rate_));
  params.emplace("weight_decay", std::to_string(weight_decay_));
  return std::accumulate(params.cbegin(), params.cend(), std::string(""),
    [](const std::string& sum, const ValueType& i) {
      return sum + '\n' + i.first + '=' + i.second;
    }).substr(1);
}
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_OPTIMIZER_HPP
