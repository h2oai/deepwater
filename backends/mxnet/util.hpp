/*!
 * Copyright (c) 2016 by Contributors
 */

#ifndef DEEPWATER_UTIL_HPP
#define DEEPWATER_UTIL_HPP

#include <vector>
#include "include/ndarray.h"
#include "include/logging.h"

std::vector<float> loadNDArray(const char * fname) {
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoad(fname, &out_size, &out_arr, &out_name_size,
                         &out_names), 0);
  CHECK_EQ(out_name_size, out_size);
  mxnet::cpp::NDArray nd_res = mxnet::cpp::NDArray(out_arr[0]);
  size_t nd_size = nd_res.Size();
  std::vector<float> res(nd_size);
  nd_res.SyncCopyToCPU(&res, nd_size);
  return res;
}

#endif
