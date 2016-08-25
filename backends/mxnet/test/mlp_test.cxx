/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <map>

#include "network_def.hpp"

using namespace mxnet::cpp;

Symbol MLPSymbol() {
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("softmax_label");
  Symbol inputdropout = Dropout("dropout0", data, 0.1);

  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc1 = FullyConnected("fc1", data, fc1_w, fc1_b, 4096);
  Symbol act1 = Activation("relu1", fc1, "relu");
  Symbol dropout1 = Dropout("dropout1", fc1, 0.5);

  Symbol fc2_w("fc2_w"), fc2_b("fc2_b");
  Symbol fc2 = FullyConnected("fc2", act1, fc2_w, fc2_b, 4096);
  Symbol act2 = Activation("relu2", fc2, "relu");
  Symbol dropout2 = Dropout("dropout2", fc2, 0.5);

  Symbol fc3_w("fc3_w"), fc3_b("fc3_b");
  Symbol fc3 = FullyConnected("fc3", act2, fc3_w, fc3_b, 4096);
  Symbol act3 = Activation("relu3", fc2, "relu");
  Symbol dropout3 = Dropout("dropout3", fc3, 0.5);

  Symbol fc4_w("fc4_w"), fc4_b("fc4_b");
  Symbol fc4 = FullyConnected("fc4", act3, fc4_w, fc4_b, 10);
  return SoftmaxOutput("softmax", fc4, data_label);
}

int main(int argc, char const *argv[]) {
  /*setup basic configs*/
  int W = 28;
  int H = 28;
  int batch_size = 256;
  int max_epoch = 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-6;

#if MSHADOW_USE_CUDA == 1
  Context ctx_dev = Context(DeviceType::kGPU, 0);
#else
  Context ctx_dev = Context(DeviceType::kCPU, 0);
#endif

  auto lenet = MLPSymbol();
  std::map<std::string, NDArray> args_map;

  args_map["data"] = NDArray(Shape(batch_size, 1, W * H), ctx_dev);
  args_map["softmax_label"] = NDArray(Shape(batch_size), ctx_dev);
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
    Accuracy train_acc;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      args_map["data"] = data_batch.data.Copy(ctx_dev);
      args_map["softmax_label"] = data_batch.label.Copy(ctx_dev);
      NDArray::WaitAll();
      auto *exec = lenet.SimpleBind(ctx_dev, args_map);
      exec->Forward(true);
      exec->Backward();
      exec->UpdateAll(&opt, learning_rate, weight_decay);
      train_acc.Update(data_batch.label, exec->outputs[0]);
      delete exec;
    }
    LG << "Training acc: " << train_acc.Get();

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      args_map["data"] = data_batch.data.Copy(ctx_dev);
      args_map["softmax_label"] = data_batch.label.Copy(ctx_dev);
      NDArray::WaitAll();
      auto *exec = lenet.SimpleBind(ctx_dev, args_map);
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
      delete exec;
    }
    LG << "Val acc: " << acu.Get();
  }
  MXNotifyShutdown();
  return 0;
}
