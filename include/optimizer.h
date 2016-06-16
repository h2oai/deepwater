/*!
*  Copyright (c) 2016 by Contributors
* \file optimizer.h
* \brief definition of optimizer
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNETCPP_OPTIMIZER_H
#define MXNETCPP_OPTIMIZER_H

#include <map>
#include <string>
#include "base.h"
#include "logging.h"
#include "ndarray.h"

namespace mxnet {
namespace cpp {

/*!
* \brief Optimizer interface
*/
class Optimizer {
 public:
  /*!
  * \brief Operator constructor, the optimizer is not initialized until the
  * first Update
  * \param opt_type type of the optimizer
  * \param learning_rate
  */
  Optimizer(const std::string &opt_type, mx_float learning_rate, mx_float weight_decay);
  /*!
  * \brief destructor, free the handle
  */
  ~Optimizer() {
    if (init_) MXOptimizerFree(handle_);
  }
  /*!
  * \brief set config parameters
  * \param name name of the config parameter
  * \param value value of the config parameter
  * \return reference of self
  */
  template <typename T>
  Optimizer &SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return *this;
  }
  /*!
  *  \brief Update a weight with gradient.
  *  \param index the unique index for the weight.
  *  \param weight the weight to update.
  *  \param grad gradient for the weight.
  */
  void Update(int index, NDArray weight, NDArray grad);
  // TODO(zhangcheng-qinyinghua)
  // implement Update a list of arrays, maybe in the form of map
  // void Update(int index, std::vector<NDArray> weights, std::vector<NDArray>
  // grad, mx_float lr);

  /*!
  *  \brief Serialize the optimizer parameters to a string.
  *  \return serialization
  */
  std::string Serialize() const;

 private:
  bool init_;
  mx_float learning_rate_, weight_decay_;
  std::string opt_type_;
  Optimizer(const Optimizer &);
  Optimizer &operator=(const Optimizer &);
  OptimizerHandle handle_;
  OptimizerCreator creator_;
  std::map<std::string, std::string> params_;
};
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_OPTIMIZER_H
