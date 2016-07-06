
#include "include/MxNetCpp.h"

using namespace mxnet::cpp;

std::vector<Symbol> lstm(int num_hidden, Symbol indata, std::vector<Symbol> prev_state,
                         std::vector<Symbol> param, int seqidx, int layeridx,
                         mx_float dropout) {
  if (dropout > 0)
    indata = Dropout("indata", indata, dropout);

  Symbol i2h = FullyConnected("t" + std::to_string(seqidx) + ".l" + std::to_string(layeridx) + ".i2h",
                              indata, param[0], param[1], num_hidden * 4);

  Symbol h2h = FullyConnected("t" + std::to_string(seqidx) + ".l" + std::to_string(layeridx) + ".h2h",
                              prev_state[1], param[2], param[3],
                              num_hidden * 4);

  Symbol gates = i2h + h2h;
  Symbol slice_gates = SliceChannel("t" + std::to_string(seqidx) + ".l" + std::to_string(layeridx) + ".slice",
                                    gates, 4);

  Symbol in_gate = Activation("in_gate", slice_gates[0], "sigmoid");

  Symbol in_transform = Activation("in_transform", slice_gates[1], "tanh");

  Symbol forget_gate = Activation("forget_gate", slice_gates[2], "sigmoid");

  Symbol out_gate = Activation("out_gate", slice_gates[3], "sigmoid");

  Symbol next_c = (forget_gate * prev_state[0]) + (in_gate * in_transform);

  Symbol next_h = out_gate * Activation("next_c", next_c, "tanh");

  std::vector<Symbol> lst;
  lst.push_back(next_c);
  lst.push_back(next_h);
  return lst;
}

int main() {
  int batch_size = 32;
  int seq_len = 2;
  int num_hidden = 16;
  int num_embed = 16;
  int num_lstm_layer = 1;
  int num_round = 1;
  mx_float learning_rate = 0.1;
  mx_float wd = 0.00001;
  int clip_gradient = 1;
  int update_period = 1;
  int num_label = 70;
  int input_size = 70;
  mx_float dropout = 0;

  Symbol embed_weight = Symbol::Variable("embed.weight");
  Symbol cls_weight = Symbol::Variable("cls.weight");
  Symbol cls_bias = Symbol::Variable("cls.bias");

  std::vector<std::vector<Symbol> > param_cells;

  std::vector<Symbol> last_hidden;

  for (int i = 1; i <= num_lstm_layer; i++) {
    std::vector<Symbol> lst; 
    lst.push_back(Symbol::Variable("l" + std::to_string(i) + ".i2h.weight"));
    lst.push_back(Symbol::Variable("l" + std::to_string(i) +  ".i2h.bias"));
    lst.push_back(Symbol::Variable());
    lst.push_back(Symbol::Variable());
    param_cells.push_back(lst);
  }

  std::vector<std::vector<Symbol> > last_states;
  for (int i = 1; i <= num_lstm_layer; i++) {
    std::vector<Symbol> lst;
    lst.push_back(Symbol::Variable("l" + std::to_string(i) + ".init.c"));
    lst.push_back(Symbol::Variable("l" + std::to_string(i) + ".init.h"));
    last_states.push_back(lst);
  }

  Symbol label = Symbol::Variable("label");
  Symbol data = Symbol::Variable("data");

  Symbol embed = Embedding("embed", data, embed_weight,
                           input_size, num_embed);

  Symbol wordvec = SliceChannel("wordvec", embed, seq_len, 1);

  for (int seqidx = 0; seqidx < seq_len; seqidx++) {
    Symbol hidden = wordvec[seqidx];
    for (int i = 0; i < num_lstm_layer; i++) {
      mx_float dp;
      if (i == 0) {
        dp = 0;
      } else {
        dp = dropout;
      }

      std::vector<Symbol> next_state = lstm(num_hidden, hidden,
                                            last_states[i],
                                            param_cells[i],
                                            seqidx + 1,
                                            i + 1, dp);

      hidden = next_state[1];
      last_states[i] = next_state;
    }
    if (dropout > 0)
      hidden = Dropout("hidden", hidden, dropout);
    
    last_hidden.push_back(hidden);

  }

  Symbol concat = Concat("concat", last_hidden, seq_len, 1);
  Symbol fc = FullyConnected("fc", concat,
                             cls_weight, cls_bias,
                             num_label);

  Symbol label2 = transpose("label2", label, label);
  Symbol label3 = Reshape("label3", label2);
  Symbol loss_all = SoftmaxOutput("sm", fc, label3);
  loss_all.Save("lstm.json");
}
