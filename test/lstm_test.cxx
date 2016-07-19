/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <cassert>
#include "../include/MxNetCpp.h"
#include "../network_def.hpp"

using namespace mxnet::cpp;

int main() {
  int batch_size = 32;
  int seq_len = 129;
  int num_hidden = 512;
  int num_embed = 256;
  int num_lstm_layer = 3;
  //int num_round = 21;
  mx_float learning_rate = 0.01;
  mx_float wd = 0.00001;
  /*int clip_gradient = 1;*/
  /*int update_period = 1;*/
  int input_size = 77 + 1;
  int num_label = input_size;

  Symbol rnn_sym = lstm_unroll(num_lstm_layer, seq_len, input_size,
                               num_hidden, num_embed, num_label, 0.0);

  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(batch_size, seq_len), Context::cpu());
  args_map["softmax_label"] = NDArray(Shape(batch_size, seq_len), Context::cpu());

  rnn_sym.InferArgsMap(Context::cpu(), &args_map, args_map);

  Optimizer * opt = new Optimizer("ccsgd", learning_rate, wd);
  opt->SetParam("momentum", 0.9);
  opt->SetParam("rescale_grad", 1.0 / batch_size);
  opt->SetParam("clip_gradient", 10);

  Executor * exec = rnn_sym.SimpleBind(Context::cpu(), args_map);

}
