
#include "include/MxNetCpp.h"

using namespace mxnet::cpp;

std::vector<Symbol> lstm(int num_hidden, Symbol indata, 
                         std::vector<Symbol> prev_state, std::vector<Symbol> param, 
                         int seqidx, int layeridx, mx_float dropout=0.0) {

  if (dropout > 0)
    indata = Dropout("dp", indata, dropout);

  Symbol i2h = FullyConnected("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_i2h",
                              indata, param[0], param[1],
                              num_hidden *4); 

  Symbol h2h = FullyConnected("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_h2h",
                              prev_state[1], param[2], param[3],
                              num_hidden * 4);

  Symbol gates = i2h + h2h;
  Symbol slice_gates = SliceChannel("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_slice",
                                    gates, 4);

  Symbol in_gate = Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_in_gates",
                              slice_gates[0], "sigmod");
  Symbol in_transform = Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_in_transform",
                                   slice_gates[1], "tanh");
  Symbol forget_gate = Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_forget_data",
                                  slice_gates[2], "sigmoid");
  Symbol out_gate = Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_out_gate",
                               slice_gates[3], "sigmoid");

  Symbol next_c = (forget_gate * prev_state[0]) + (in_gate * in_transform);
  Symbol next_h = out_gate * Activation("", next_c, "tanh");
  std::vector<Symbol> state;
  state.push_back(next_c);
  state.push_back(next_h);
  return state;
}

Symbol lstm_unroll(int num_lstm_layer, int seq_len, int input_size,
                   int num_hidden, int num_embed, int num_label,
                   mx_float dropout=0.0) {
  Symbol embed_weight = Symbol::Variable("embed_weight");
  Symbol cls_weight = Symbol::Variable("cls_weight");
  Symbol cls_bias = Symbol::Variable("cls_bias");

  std::vector<std::vector<Symbol>> param_cells;
  std::vector<std::vector<Symbol>> last_states;

  for (int i = 0; i < num_lstm_layer; i++) {
    std::vector<Symbol> param;
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_i2h_weight"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_i2h_bias"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_h2h_weight"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_h2h_bias"));
    param_cells.push_back(param);

    std::vector<Symbol> state;
    state.push_back(Symbol::Variable("l" + std::to_string(i) + "_init_c"));
    state.push_back(Symbol::Variable("l" + std::to_string(i) + "_init_h"));
    last_states.push_back(state);
  }

  Symbol label = Symbol::Variable("label");
  std::vector<Symbol> last_hidden;
  for (int seqidx = 0; seqidx < seq_len; seqidx++) {
    Symbol data = Symbol::Variable("t" + std::to_string(seqidx) + "_data");
    Symbol hidden = Embedding("t" + std::to_string(seqidx) + "_embed",
                              data, embed_weight, input_size, num_embed);
    for (int i = 0; i < num_lstm_layer; i++) {
      mx_float dp;
      if (i ==0) 
        dp = 0.0;
      else
        dp = dropout;

      std::vector<Symbol> next_state = lstm(num_hidden, hidden, last_states[i],
                                            param_cells[i], seqidx, i, dp);
      hidden = next_state[1];
      last_states[i] = next_state;
    }

    if (dropout > 0)
      hidden = Dropout("", hidden, dropout);

    last_hidden.push_back(hidden);
  }
  Symbol concat = Concat("", last_hidden, last_hidden.size(), 0);
  Symbol fc = FullyConnected("", concat, cls_weight, cls_bias, num_label);
  Symbol sm = SoftmaxOutput("sm", fc, label);
  std::vector<Symbol> list_all;
  list_all.push_back(sm);
  for (int i = 0; i < num_lstm_layer; i++) {
    last_states[i][0] = BlockGrad("l" + std::to_string(i) + "_last_c", last_states[i][0]);
    last_states[i][1] = BlockGrad("l" + std::to_string(i) + "_last_c", last_states[i][1]);
  }

  for (size_t i = 0; i < last_states.size(); i++) {
    list_all.push_back(last_states[i][0]);
  }

  for (size_t i = 0; i < last_states.size(); i++) {
    list_all.push_back(last_states[i][1]);
  }

  return Symbol::Group(list_all);
}

int main() {
  int batch_size = 32;
  int seq_len = 32;
  int num_hidden = 256;
  int num_embed = 256;
  int num_lstm_layer = 2;
  int num_round = 21;
  mx_float learning_rate = 0.01;
  mx_float wd = 0.00001;
  int clip_gradient = 1;
  int update_period = 1;
  int input_size = 63;
  int num_label = input_size;
  Symbol rnn_sym = lstm_unroll(num_lstm_layer, num_hidden, seq_len, input_size,
                               num_embed, num_label, 0.0);
}
