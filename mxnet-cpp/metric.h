/*!
*  Copyright (c) 2016 by Contributors
* \file base.h
* \brief metrics defined
* \author Zhang Chen
*/

#ifndef MXNETCPP_METRIC_H
#define MXNETCPP_METRIC_H

#include <string>
#include <vector>
#include "ndarray.h"
#include "logging.h"

namespace mxnet {
namespace cpp {

class EvalMetric {
 public:
  explicit EvalMetric(const std::string& name, int num = 0) : name(name), num(num) {}
  virtual void Update(NDArray labels, NDArray preds) = 0;
  void Reset();
  float Get() { return sum_metric / num_inst; }
  void GetNameValue();

 protected:
  std::string name;
  int num;
  float sum_metric = 0.0f;
  int num_inst = 0;
};

class Accuracy : public EvalMetric {
 public:
  Accuracy() : EvalMetric("accuracy") {}

  void Update(NDArray labels, NDArray preds) {
    CHECK_EQ(labels.GetShape().size(), 1);
    mx_uint len = labels.GetShape()[0];
    std::vector<mx_float> pred_data(len);
    std::vector<mx_float> label_data(len);
    preds.ArgmaxChannel().SyncCopyToCPU(&pred_data, len);
    labels.SyncCopyToCPU(&label_data, len);
    NDArray::WaitAll();
    for (mx_uint i = 0; i < len; ++i) {
      sum_metric += (pred_data[i] == label_data[i]) ? 1 : 0;
      num_inst += 1;
    }
  }
};

}  // namespace cpp
}  // namespace mxnet

#endif /* end of include guard: MXNETCPP_METRIC_H */

