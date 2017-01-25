/*!
 *  Copyright (c) 2016 by Contributors
 * \file executor.cxx
 * \brief implementation of the executor
 * \author Zhang Chen, Chuntao Hong
 */

#include <vector>
#include <set>
#include <map>
#include <string>
#include "executor.h"
#include "optimizer.h"

namespace mxnet {
namespace cpp {
Executor::Executor(const Symbol &symbol, Context context,
                   const std::vector<NDArray> &arg_arrays,
                   const std::vector<NDArray> &grad_arrays,
                   const std::vector<OpReqType> &grad_reqs,
                   const std::vector<NDArray> &aux_arrays,
                   const std::map<std::string, Context> &group_to_ctx,
                   Executor *shared_exec) {
  this->arg_arrays = arg_arrays;
  this->grad_arrays = grad_arrays;
  this->aux_arrays = aux_arrays;
  this->symbol_ = symbol;

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

  std::vector<const char *> map_keys;
  std::vector<int> dev_types, dev_ids;
  for (const auto &s : group_to_ctx) {
    map_keys.push_back(s.first.c_str());
    dev_types.push_back(s.second.GetDeviceType());
    dev_ids.push_back(s.second.GetDeviceId());
  }

  ExecutorHandle *shared_exec_handle =
      shared_exec == nullptr ? nullptr : &shared_exec->handle_;

  CHECK_EQ(MXExecutorBindEX(symbol.GetHandle(), context.GetDeviceType(),
                            context.GetDeviceId(), group_to_ctx.size(),
                            map_keys.data(), dev_types.data(), dev_ids.data(),
                            arg_handles.size(), arg_handles.data(),
                            grad_handles.data(), grad_reqs_uint.data(),
                            aux_handles.size(), aux_handles.data(),
                            shared_exec_handle, &handle_),
           0);

  mx_uint out_size;
  NDArrayHandle *out_array;
  CHECK_EQ(MXExecutorOutputs(handle_, &out_size, &out_array), 0);
  for (mx_uint i = 0; i < out_size; ++i) {
    outputs.push_back(NDArray(out_array[i]));
  }
}

std::string Executor::DebugStr() {
  const char *output;
  MXExecutorPrint(handle_, &output);
  return std::string(output);
}

void Executor::UpdateAll(Optimizer *opt, float lr, float wd,
                         int arg_update_begin, int arg_update_end) {
  arg_update_end = arg_update_end < 0 ? arg_arrays.size() - 1 : arg_update_end;
  for (int i = arg_update_begin; i < arg_update_end; ++i) {
    opt->Update(i, arg_arrays[i], grad_arrays[i], lr, wd);
  }
}

std::map<std::string, NDArray> Executor::GetDict(const std::vector<std::string> &names,
                                                 const std::vector<NDArray> &arrays) {
  std::map<std::string, NDArray> ret;
  std::set<std::string> name_set;
  for (const auto &s : names) {
    CHECK_EQ(name_set.find(s), name_set.end()) << "Duplicate names detected, "
        << s;
    name_set.insert(s);
  }
  CHECK_EQ(name_set.size(), arrays.size())
      << "names size not equal to arrays size";
  for (size_t i = 0; i < names.size(); ++i) {
    ret[names[i]] = arrays[i];
  }
  return ret;
}
}  // namespace cpp
}  // namespace mxnet

