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

 private:
  int bucket, batch_size;
  size_t start_idx;
  std::vector<mx_float> data, label;
  std::vector<int> bucket_idx_all;
  std::vector<std::vector<mx_float> > all_data;
};

TextIter::TextIter(const std::string & path,
                   std::map<char, int> vocab,
                   int bucket,
                   int batch_size): bucket(bucket), batch_size(batch_size) {
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
  while (all_data.size() > tmp) {
    all_data.pop_back();
  }
  bucket_idx_all.clear();
  for (size_t i = 0; i < all_data.size(); i++) {
    bucket_idx_all.push_back(i);
  }
  std::random_shuffle(bucket_idx_all.begin(), bucket_idx_all.end());
}

void TextIter::Reset() {
  start_idx = 0;
}

bool TextIter::Next() {
  data.clear();
  label.clear();
  if (start_idx >= all_data.size())
    return false;

  for (size_t i = start_idx; i < start_idx + batch_size; i++) {
    int idx = bucket_idx_all[i]; 
    std::vector<mx_float> data_tmp = all_data[idx];
    std::vector<mx_float> label_tmp = all_data[idx];
    label_tmp.erase(label_tmp.begin());
    label_tmp.push_back(0.0);
    assert(label_tmp.size() == data_tmp.size());
    data.insert(data.end(), data_tmp.begin(), data_tmp.end());
    label.insert(label.end(), label_tmp.begin(), label_tmp.end());
  }
  return true;
}

int main() {
  std::vector<std::string> sentences = readContent("./obama.txt");
  std::string content;
  for (size_t i = 0; i < sentences.size(); i++) {
    content += sentences[i] + "\n";
  }
  std::map<char, int> vocab = buildVocab(content);
  int batch_size = 32;
  int seq_len = 129;
  int num_hidden = 512;
  int num_embed = 256;
  int num_lstm_layer = 3;
  int num_round = 21;
  mx_float learning_rate = 0.01;
  mx_float wd = 0.00001;
  int clip_gradient = 1;
  int update_period = 1;
  int input_size = vocab.size() + 1;
  int num_label = input_size;
  TextIter iter("./obama.txt", vocab, seq_len, batch_size);

  Symbol rnn_sym = lstm_unroll(num_lstm_layer, seq_len, input_size,
                               num_hidden, num_embed, num_label, 0.0);

  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(batch_size, 1, seq_len), Context::cpu());
  args_map["softmax_label"] = NDArray(Shape(batch_size, 1, seq_len), Context::cpu());

  rnn_sym.InferArgsMap(Context::cpu(), &args_map, args_map);

  Optimizer * opt = new Optimizer("ccsgd", learning_rate, wd);
  opt->SetParam("momentum", 0.9);
  opt->SetParam("rescale_grad", 1.0 / batch_size);
  opt->SetParam("clip_gradient", 10);

  Executor * exec = rnn_sym.SimpleBind(Context::cpu(), args_map);
  iter.Reset();
  iter.Next();
  std::vector<mx_float> data = iter.getData();
  std::vector<mx_float> label = iter.getLabel();
  NDArray data_n = NDArray(data.data(), Shape(batch_size, 1, seq_len), Context::gpu());

  NDArray label_n = NDArray(label.data(), Shape(batch_size, 1, seq_len), Context::gpu());
  data_n.CopyTo(&args_map["data"]);
  label_n.CopyTo(&args_map["softmax_label"]);
  NDArray::WaitAll();
  exec->Forward(true);
  exec->Backward();
  exec->UpdateAll(opt, learning_rate, wd);
  NDArray::WaitAll();

  std::vector<float> preds(batch_size * seq_len);
  exec->outputs[0].SyncCopyToCPU(&preds, batch_size * seq_len);
  NDArray::WaitAll();

}
