/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <map>

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

#ifdef GPU
  Context ctx_dev = Context(DeviceType::kGPU, 0);
#else
  Context ctx_dev = Context(DeviceType::kCPU, 0);
#endif

  args_map["data"] = NDArray(Shape(batch_size, 1, W, H), ctx_dev);
  args_map["data_label"] = NDArray(Shape(batch_size), ctx_dev);
  lenet.InferArgsMap(ctx_dev, &args_map, args_map);

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

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      args_map["data"] = data_batch.data.Copy(ctx_dev);
      args_map["data_label"] = data_batch.label.Copy(ctx_dev);
      NDArray::WaitAll();
      auto * exec = lenet.SimpleBind(ctx_dev, args_map);
      exec->Forward(true);
      exec->Backward();
      exec->UpdateAll(&opt, learning_rate, weight_decay);
      delete exec;
    }

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      args_map["data"] = data_batch.data.Copy(ctx_dev);
      args_map["data_label"] = data_batch.label.Copy(ctx_dev);
      NDArray::WaitAll();
      auto * exec = lenet.SimpleBind(ctx_dev, args_map);
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
      delete exec;
    }
    LG << "Accuracy: " << acu.Get();
  }
  return 0;
}
