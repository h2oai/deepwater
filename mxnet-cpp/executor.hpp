/*!
 *  Copyright (c) 2016 by Contributors
 * \file executor.hpp
 * \brief implementation of the executor
 * \author Zhang Chen, Chuntao Hong
 */

#ifndef MXNETCPP_EXECUTOR_HPP
#define MXNETCPP_EXECUTOR_HPP

#include <vector>
#include "executor.h"
#include "optimizer.h"

namespace mxnet {
namespace cpp {
Executor::Executor(const Symbol &symbol, Context context,
                   const std::vector<NDArray> &arg_arrays,
                   const std::vector<NDArray> &grad_arrays,
                   const std::vector<OpReqType> &grad_reqs,
                   const std::vector<NDArray> &aux_arrays) {
  this->arg_arrays = arg_arrays;
  this->grad_arrays = grad_arrays;
  this->aux_arrays = aux_arrays;

  std::vector<NDArrayHandle> arg_handles;
  std::vector<NDArrayHandle> grad_handles;
  std::vector<NDArrayHandle> aux_handles;

  for (const auto &array : arg_arrays) {
    arg_handles.push_back(array.GetHandle());
  }
  for (const auto &array : grad_arrays) {
    grad_handles.push_back(array.GetHandle());
  }
  for (const auto &array : aux_arrays) {
    aux_handles.push_back(array.GetHandle());
  }

  std::vector<mx_uint> grad_reqs_uint;
  for (auto s : grad_reqs) grad_reqs_uint.push_back(s);

  CHECK_EQ(MXExecutorBind(symbol.GetHandle(), context.GetDeviceType(),
                          context.GetDeviceId(), arg_handles.size(),
                          arg_handles.data(), grad_handles.data(),
                          grad_reqs_uint.data(), aux_handles.size(),
                          aux_handles.data(), &handle_),
           0);

  mx_uint out_size;
  NDArrayHandle *out_array;
  CHECK_EQ(MXExecutorOutputs(handle_, &out_size, &out_array), 0);
  for (mx_uint i = 0; i < out_size; ++i) {
    outputs.push_back(NDArray(out_array[i]));
  }
}

void Executor::UpdateAll(Optimizer *opt, float lr, float wd,
                         int arg_update_begin, int arg_update_end) {
  arg_update_end = arg_update_end < 0 ? arg_arrays.size() - 1 : arg_update_end;
  for (int i = arg_update_begin; i < arg_update_end; ++i) {
    opt->Update(i, arg_arrays[i], grad_arrays[i]);
  }
}
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_EXECUTOR_HPP
