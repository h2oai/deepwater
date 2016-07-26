/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <map>

#include "../network_def.hpp"

using namespace mxnet::cpp;

int main(int argc, char const *argv[]) {
  int batch_size = 40;
  int max_epoch = 100;
  float learning_rate = 0.01;
  float weight_decay = 1e-4;

  auto inception_bn_net = InceptionSymbol(10);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

#ifdef GPU
  Context ctx_dev = Context(DeviceType::kGPU, 0);
#else
  Context ctx_dev = Context(DeviceType::kCPU, 0);
#endif

  args_map["data"] = NDArray(Shape(batch_size, 3, 224, 224), ctx_dev);
  args_map["data_label"] = NDArray(Shape(batch_size), ctx_dev);
  inception_bn_net.InferArgsMap(ctx_dev, &args_map, args_map);

  auto train_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./sf1_train.lst")
      .SetParam("path_imgrec", "./sf1_train.rec")
      .SetParam("data_shape", Shape(3, 224, 224))
      .SetParam("batch_size", batch_size)
      .SetParam("shuffle", 1)
      .CreateDataIter();

  auto val_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./sf1_val.lst")
      .SetParam("path_imgrec", "./sf1_val.rec")
      .SetParam("data_shape", Shape(3, 224, 224))
      .SetParam("batch_size", batch_size)
      .CreateDataIter();

  Optimizer opt("ccsgd", learning_rate, weight_decay);
  opt.SetParam("momentum", 0.9)
      .SetParam("rescale_grad", 1.0 / batch_size)
      .SetParam("clip_gradient", 10);

  auto * exec = inception_bn_net.SimpleBind(ctx_dev, args_map);

  NDArray::Load("/home/ops/Inception/Inception_BN-0039.params", nullptr, &args_map);

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();

      exec->Forward(true);
      exec->Backward();
      exec->UpdateAll(&opt, learning_rate, weight_decay);
      NDArray::WaitAll();
    }

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Accuracy: " << acu.Get();
  }
  NDArray::Save("new_inception_bn.params", args_map);
  delete exec;
  return 0;
}
