/*!
*  Copyright (c) 2016 by Contributors
* \file operator.hpp
* \brief implementation of operator
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNETCPP_OPERATOR_HPP
#define MXNETCPP_OPERATOR_HPP

#include <string>
#include <vector>
#include "base.h"
#include "op_map.h"
#include "operator.h"

namespace mxnet {
namespace cpp {
OpMap *Operator::op_map_ = new OpMap();

Operator::Operator(const std::string &operator_name) {
  handle_ = op_map_->GetSymbolCreator(operator_name);
}

Symbol Operator::CreateSymbol(const std::string &name) {
  const char *pname = name == "" ? nullptr : name.c_str();

  SymbolHandle symbol_handle;
  std::vector<const char *> input_keys;
  std::vector<const char *> param_keys;
  std::vector<const char *> param_values;

  for (auto &data : params_) {
    param_keys.push_back(data.first.c_str());
    param_values.push_back(data.second.c_str());
  }
  for (auto &data : this->input_keys) {
    input_keys.push_back(data.c_str());
  }
  const char **input_keys_p =
      (input_keys.size() > 0) ? input_keys.data() : nullptr;

  MXSymbolCreateAtomicSymbol(handle_, param_keys.size(), param_keys.data(),
                             param_values.data(), &symbol_handle);
  MXSymbolCompose(symbol_handle, pname, input_values.size(), input_keys_p,
                  input_values.data());
  return Symbol(symbol_handle);
}
Operator &Operator::SetInput(const std::string &name, Symbol symbol) {
  input_keys.push_back(name.c_str());
  input_values.push_back(symbol.GetHandle());
  return *this;
}
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_OPERATOR_HPP
