/*!
*  Copyright (c) 2016 by Contributors
* \file operator.h
* \brief definition of operator
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNETCPP_OPERATOR_H
#define MXNETCPP_OPERATOR_H

#include <map>
#include <string>
#include <vector>
#include "base.h"
#include "op_map.h"
#include "symbol.h"

namespace mxnet {
namespace cpp {
class Mxnet;
/*!
* \brief Operator interface
*/
class Operator {
 public:
  /*!
  * \brief Operator constructor
  * \param operator_name type of the operator
  */
  explicit Operator(const std::string &operator_name);
  Operator &operator=(const Operator &rhs);
  /*!
  * \brief set config parameters
  * \param name name of the config parameter
  * \param value value of the config parameter
  * \return reference of self
  */
  template <typename T>
  Operator &SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return *this;
  }
  /*!
  * \brief add an input symbol
  * \param name name of the input symbol
  * \param symbol the input symbol
  * \return reference of self
  */
  Operator &SetInput(const std::string &name, Symbol symbol);
  /*!
  * \brief add an input symbol
  * \param symbol the input symbol
  */
  void PushInput(const Symbol &symbol) {
    input_values.push_back(symbol.GetHandle());
  }
  /*!
  * \brief add input symbols
  */
  template <class T, class... Args>
  void PushInput(const Symbol &symbol, const T &t, Args... args) {
    PushInput(symbol);
    PushInput(t, args...);
  }
  /*!
  * \brief add input symbols
  * \return reference of self
  */
  Operator &operator()() { return *this; }
  /*!
  * \brief add input symbols
  * \param symbol the input symbol
  * \return reference of self
  */
  Operator &operator()(const Symbol &symbol) {
    input_values.push_back(symbol.GetHandle());
    return *this;
  }
  /*!
  * \brief add a list of input symbols
  * \param symbols the vector of the input symbols
  * \return reference of self
  */
  Operator &operator()(const std::vector<Symbol> &symbols) {
    for (auto &s : symbols) {
      input_values.push_back(s.GetHandle());
    }
    return *this;
  }
  /*!
  * \brief add input symbols
  * \return reference of self
  */
  template <typename T, typename... Args>
  Operator &operator()(const Symbol &symbol, const T &t, Args... args) {
    PushInput(symbol, t, args...);
    return *this;
  }
  /*!
  * \brief create a Symbol from the current operator
  * \param name the name of the operator
  * \return the operator Symbol
  */
  Symbol CreateSymbol(const std::string &name = "");

 private:
  std::map<std::string, std::string> params_desc_;
  bool variable_params_ = false;
  std::map<std::string, std::string> params_;
  std::vector<SymbolHandle> input_values;
  std::vector<std::string> input_keys;
  AtomicSymbolCreator handle_;
  static OpMap *op_map_;
};
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_OPERATOR_H
