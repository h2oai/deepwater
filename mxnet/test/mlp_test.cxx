/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <map>

#include "initializer.h"
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
  Symbol act3 = Activation("relu3", fc3, "relu");
  Symbol dropout3 = Dropout("dropout3", fc3, 0.5);

  Symbol fc4_w("fc4_w"), fc4_b("fc4_b");
  Symbol fc4 = FullyConnected("fc4", act3, fc4_w, fc4_b, 10);
  return SoftmaxOutput("softmax", fc4, data_label);
}

int main(int argc, char const *argv[]) {
  int batch_size = 128;
  int W = 28;
  int H = 28;
  int channels = 1;
  Shape shape(batch_size, channels, W, H);
  int max_epoch = 100;
  float learning_rate = 1e-2;
  float weight_decay = 1e-6;
  float momentum = 0.9;
  float clip_gradient = 10;

  MXRandomSeed(42);
  auto net = MLPSymbol();

  //net.Save("/tmp/mx.json");
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

#if MSHADOW_USE_CUDA == 0
  Context ctx_dev = Context(DeviceType::kCPU, 0);
#else
  Context ctx_dev = Context(DeviceType::kGPU, 0);
#endif

  args_map["data"] = NDArray(shape, ctx_dev);
  args_map["softmax_label"] = NDArray(Shape(batch_size), ctx_dev);
  net.InferArgsMap(ctx_dev, &args_map, args_map);

  auto train_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./train-images-idx3-ubyte")
      .SetParam("label", "./train-labels-idx1-ubyte")
      .SetParam("data_shape", shape)
      .SetParam("batch_size", batch_size)
      .SetParam("shuffle", 0)
      .CreateDataIter();

  auto val_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./t10k-images-idx3-ubyte")
      .SetParam("label", "./t10k-labels-idx1-ubyte")
      .SetParam("data_shape", shape)
      .SetParam("batch_size", batch_size)
      .CreateDataIter();

  Optimizer *opt = OptimizerRegistry::Find("ccsgd");
  opt->SetParam("momentum", momentum)
      ->SetParam("rescale_grad", 1.0 / batch_size)
      ->SetParam("clip_gradient", clip_gradient);

  auto * exec = net.SimpleBind(ctx_dev, args_map);
  args_map = exec->arg_dict();

  Xavier xavier = Xavier(Xavier::gaussian, Xavier::in, 2.34);
  for (auto &arg : args_map) {
    xavier(arg.first, &arg.second);
  }

  aux_map = exec->aux_dict();
  for (auto &aux : aux_map) {
    xavier(aux.first, &aux.second);
  }

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
      exec->UpdateAll(opt, learning_rate, weight_decay);
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
  }
  delete exec;
  MXNotifyShutdown();
  return 0;
}
