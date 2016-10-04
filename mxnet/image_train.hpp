/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __H2O_IMAGE_TRAIN_H__
#define __H2O_IMAGE_TRAIN_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "network_def.hpp"

class ImageTrain{
  private:
    void setClassificationDimensions(int num_classes, int batch_size);
    void setOptimizer();
    void initializeState();
 public:
  explicit ImageTrain(int w = 0, int h = 0, int c = 0,
  			int device = 0, int seed = 0, bool gpu = true);
  void buildNet(int num_classes, int batch_size, char * net_name,
		 int num_hidden=0,
                 int *hidden=nullptr,
                 char**activations=nullptr,
                 double input_dropout=0,
                 double *hidden_dropout=nullptr);

  void setSeed(int seed);
  void loadModel(char * model_path);
  void saveModel(char * model_path);
  void loadParam(char * param_path);
  void saveParam(char * param_path);
  const char * toJson();
  std::vector<float> loadMeanImage(const char * file_path);

  void setLR(float lr) {learning_rate = lr;}
  void setWD(float wd) {weight_decay = wd;}
  void setMomentum(float m) {
    momentum = m;
    if (opt.get() != nullptr) opt->SetParam("momentum", momentum);
  }
  void setClipGradient(float c) {
    clip_gradient = c;
    if (opt.get() != nullptr) opt->SetParam("clip_gradient", clip_gradient);
  }

  //train/predict on a mini-batch
  std::vector<float> train(float * data, float * label);
  std::vector<float> predict(float * data, float * label);
  std::vector<float> predict(float * data);

 private:
  int width, height, channels;
  int batch_size, num_classes;
  float learning_rate, weight_decay, momentum, clip_gradient;

  std::map<std::string, mxnet::cpp::NDArray> args_map;
  std::map<std::string, mxnet::cpp::NDArray> aux_map;
  mxnet::cpp::Symbol mxnet_sym;
  std::unique_ptr<mxnet::cpp::Executor> exec;
  std::unique_ptr<mxnet::cpp::Optimizer> opt;
  bool is_built;
  mxnet::cpp::Context ctx_dev;

  std::vector<float> execute(float * data, float * label, bool is_train);

  // prediction probs
  std::vector<float> preds;
  mxnet::cpp::Shape shape;
};

#endif
