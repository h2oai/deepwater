/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include "../include/MxNetCpp.h"
#include "../network_def.hpp"

using namespace mxnet::cpp;

std::vector<std::string> readContent(const std::string & path) {
  std::ifstream file(path.c_str());
  std::string str;
  std::vector<std::string> res; 
  while (std::getline(file, str)) {
    res.push_back(str);
  }
  return res;
}

std::map<char, int> buildVocab(const std::string & content) {
  std::map<char, int> vocab;
  int idx = 1;
  for (size_t i = 0; i < content.length(); i++) {
    if (vocab[content.at(i)] == 0) {
      vocab[content.at(i)] = idx;
      idx++;
    }
  }
  return vocab;
}

std::vector<mx_float> text2id(const std::string & sentence,
                              const std::map<char, int> &vocab) {
  std::vector<mx_float> id;
  for (size_t i = 0; i < sentence.length(); i++) {
    id.push_back(vocab.at(sentence.at(i)));
  }
  return id;
}

bool endswith(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(),
                                    ending));
  } else {
    return false;
  }
}

bool is_param_name(const std::string & name) {
  return endswith(name, "weight") ||
      endswith(name, "bias") ||
      endswith(name, "gamma") ||
      endswith(name, "beta");
}

class TextIter {
 public:
  TextIter(const std::string & path, std::map<char, int> vocab, int bucket, int batch_size);
  void Reset();
  bool Next();
  std::vector<mx_float> getData() {return data;}
  std::vector<mx_float> getLabel() {return label;}
  std::vector<std::vector<mx_float> > all_data;

 private:
  int bucket, batch_size, start_idx;
  bool has_next;
  std::vector<mx_float> data, label;
  std::vector<int> bucket_plan;
  std::vector<int> bucket_idx_all;
};

TextIter::TextIter(const std::string & path, std::map<char, int> vocab,
                   int bucket, int batch_size): bucket(bucket), batch_size(batch_size) {
  std::vector<std::string> content = readContent(path);
  for (size_t i = 0; i < content.size(); i++) {
    if (content[i].length() <= (size_t)bucket) {
      std::vector<mx_float> tmp = text2id(content[i], vocab);
      while(tmp.size() < (size_t)bucket) {tmp.push_back(0);}
      all_data.push_back(tmp);
    }
  }
  std::cout << "Bucket: " << bucket << "; Samples: " << all_data.size()  << std::endl;
  size_t tmp = all_data.size() / batch_size * batch_size;
  bucket_plan.resize(all_data.size() / batch_size);
  std::fill(bucket_plan.begin(), bucket_plan.end(), 0);
  while (all_data.size() > tmp) {
    all_data.pop_back();
  }
  bucket_idx_all.clear();
  for (size_t i = 0; i < all_data.size(); i++) {
    bucket_idx_all.push_back(i);
  }
  std::random_shuffle(bucket_idx_all.begin(), bucket_idx_all.end());

  /*for (int i_bucket : bucket_plan) {*/
  /*}*/
}

void TextIter::Reset() {
  start_idx = 0;
  has_next = true;
}

bool TextIter::Next() {
  if (!has_next) return false;


  return true;
}

int main() {
  std::vector<std::string> sentences = readContent("./obama.txt");
  std::string content;
  for (size_t i = 0; i < sentences.size(); i++) {
    content += sentences[i] + "\n";
  }
  std::map<char, int> vocab = buildVocab(content);
  /*for (auto iter = vocab.begin(); iter != vocab.end(); iter++) {*/
  //std::cout << iter->first << " " << iter->second << std::endl;
  /*}*/
  int batch_size = 32;
  int seq_len = 129;
  int num_hidden = 256;
  int num_embed = 256;
  int num_lstm_layer = 2;
  int num_round = 21;
  mx_float learning_rate = 0.01;
  mx_float wd = 0.00001;
  int clip_gradient = 1;
  int update_period = 1;
  int input_size = 65;
  int num_label = input_size;
  TextIter iter("./obama.txt", vocab, seq_len, batch_size);

  Context ctx_dev(DeviceType::kGPU, 0);

  Symbol rnn_sym = lstm_unroll(num_lstm_layer, seq_len, input_size,
                               num_hidden, num_embed, num_label, 0.0);

  std::vector<std::string> arg_names = rnn_sym.ListArguments();

  std::map<std::string, std::vector<mx_uint> > input_shapes;
  /*
     for (size_t i = 0; i < arg_names.size(); i++) {
     if (endswith(arg_names[i], "init_c") || endswith(arg_names[i], "init_h")) {
     std::vector<mx_uint> tmp; tmp.push_back(batch_size); tmp.push_back(num_hidden);
     input_shapes[arg_names[i]] = tmp; 
     } else if (endswith(arg_names[i], "data")) {
     std::vector<mx_uint> tmp; tmp.push_back(batch_size);
     input_shapes[arg_names[i]] = tmp;
     }
     }

     std::vector<std::vector<mx_uint> > arg_shapes, aux_shapes, out_shapes;
     rnn_sym.InferShape(input_shapes, &arg_shapes, &aux_shapes, &out_shapes);
     for (size_t i = 0; i < arg_shapes.size(); i++) {
     for (size_t j = 0; j < arg_shapes[i].size(); j++) {
     std::cout << "arg_shapes " << arg_shapes[i][j] << ", ";
     }
     std::cout << std::endl;
     }
     for (size_t i = 0; i < out_shapes.size(); i++) {
     for (size_t j = 0; j < out_shapes[i].size(); j++) {
     std::cout << out_shapes[i][j] << ", ";
     }
     std::cout << std::endl;
     }
     std::vector<NDArray> arg_arrays;
     for (size_t i = 0; i < arg_shapes.size(); i++) {
     NDArray tmp(Shape(arg_shapes[i]), ctx_dev, false);
     tmp = (mx_float)0.0;
     arg_arrays.push_back(tmp);
     }

     std::cout << arg_names.size() << " " << arg_shapes.size() << std::endl;
     std::vector<NDArray> args_grad;
     for (size_t i = 0; i < arg_shapes.size(); i++) {
     if (is_param_name(arg_names[i])) {
     NDArray tmp(Shape(arg_shapes[i]), ctx_dev, false);
     tmp = (mx_float)0.0; 
     args_grad.push_back(tmp);
     }
     }
     std::cout << __LINE__ << std::endl;
     std::vector<OpReqType> grad_req_type;
     for (size_t i = 0; i < args_grad.size(); i++)
     grad_req_type.push_back(kAddTo);

     std::vector<NDArray> aux_states;
     Executor * exe = new Executor(rnn_sym, ctx_dev, arg_arrays, args_grad,
     grad_req_type, aux_states);*/
}
