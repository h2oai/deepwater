/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <map>

#include "initializer.h"
#include "network_def.hpp"

using namespace mxnet::cpp;

int main(int argc, char const *argv[]) {
  /*setup basic configs*/
  int W = 28;
  int H = 28;
  int batch_size = 128;
  int max_epoch = 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  auto lenet = LenetSymbol(10);
  std::map<std::string, NDArray> args_map;

#if MSHADOW_USE_CUDA == 0
  Context ctx_dev = Context(DeviceType::kCPU, 0);
#else
  Context ctx_dev = Context(DeviceType::kGPU, 0);
#endif

  args_map["data"] = NDArray(Shape(batch_size, 1, W, H), ctx_dev);
  args_map["softmax_label"] = NDArray(Shape(batch_size), ctx_dev);

  lenet.InferArgsMap(ctx_dev, &args_map, args_map);

  args_map["fc1_weight"] = NDArray(Shape(500, 4 * 4 * 50), ctx_dev);
  NDArray::SampleGaussian(0, 1, &args_map["fc1_weight"]);
  args_map["fc2_bias"] = NDArray(Shape(10), ctx_dev);
  args_map["fc2_bias"] = 0;

  auto train_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./train-images-idx3-ubyte")
      .SetParam("label", "./train-labels-idx1-ubyte")
      .SetParam("batch_size", batch_size)
      .SetParam("shuffle", 1)
      .SetParam("flat", 0)
      .CreateDataIter();

  auto val_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./t10k-images-idx3-ubyte")
      .SetParam("label", "./t10k-labels-idx1-ubyte")
      .CreateDataIter();

  Optimizer opt("ccsgd", learning_rate, weight_decay);
  opt.SetParam("momentum", 0.9)
      .SetParam("rescale_grad", 1.0)
      .SetParam("clip_gradient", 10);

  auto * exec = lenet.SimpleBind(ctx_dev, args_map);

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    Accuracy train_acc;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["softmax_label"]);
      NDArray::WaitAll();
      exec->Forward(true);
      exec->Backward();
      exec->UpdateAll(&opt, learning_rate, weight_decay);
      NDArray::WaitAll();
      train_acc.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Training Acc: " << train_acc.Get();

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["softmax_label"]);
      NDArray::WaitAll();
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Val Acc: " << acu.Get();
    learning_rate *= 0.98;
  }
  delete exec;
  MXNotifyShutdown();
  return 0;
}
