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
  int batch_size = 64;
  int max_epoch = 100;
  float learning_rate = 0.01;
  float weight_decay = 1e-4;

  MXRandomSeed(42);
  auto inception_bn_net = InceptionSymbol(10);

  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

#if MSHADOW_USE_CUDA == 0
  Context ctx_dev = Context(DeviceType::kCPU, 0);
#else
  Context ctx_dev = Context(DeviceType::kGPU, 0);
#endif

  args_map["data"] = NDArray(Shape(batch_size, 3, 224, 224), ctx_dev);
  args_map["softmax_label"] = NDArray(Shape(batch_size), ctx_dev);
  inception_bn_net.InferArgsMap(ctx_dev, &args_map, args_map);

  auto train_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./sf1_train.lst")
      .SetParam("path_imgrec", "./sf1_train.rec")
      .SetParam("mean_img", "mean.bin")
      .SetParam("data_shape", Shape(3, 224, 224))
      .SetParam("batch_size", batch_size)
      .SetParam("rand_crop", 1)
      .SetParam("rand_mirror", 1)
      .SetParam("shuffle", 1)
      .SetParam("max_rotate_angle", "10")
      .CreateDataIter();

  auto val_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./sf1_val.lst")
      .SetParam("path_imgrec", "./sf1_val.rec")
      .SetParam("mean_img", "mean.bin")
      .SetParam("data_shape", Shape(3, 224, 224))
      .SetParam("batch_size", batch_size)
      .CreateDataIter();

  Optimizer opt("ccsgd", learning_rate, weight_decay);
  opt.SetParam("momentum", 0.9)
      .SetParam("rescale_grad", 1.0 / batch_size)
      .SetParam("clip_gradient", 10);

  auto * exec = inception_bn_net.SimpleBind(ctx_dev, args_map);
  args_map = exec->arg_dict();
  aux_map = exec->aux_dict();

  std::map<std::string, NDArray> parameters;
  NDArray::Load("./Inception/model.params", nullptr, &parameters);

  for (const auto &k : parameters) {
    if (k.first.substr(0, 4) == "aux:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      aux_map[name] = k.second.Copy(ctx_dev);
    }
    if (k.first.substr(0, 4) == "arg:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      args_map[name] = k.second.Copy(ctx_dev);
    }
  }
  NDArray::WaitAll();


  for (int iter = 0; iter < max_epoch; ++iter) {
    Accuracy train_acc;
    LG << "Epoch: " << iter;
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
      //LG << "Training Acc: " << train_acc.Get();
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
  }
  delete exec;
  return 0;
}
