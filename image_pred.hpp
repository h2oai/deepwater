/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __DEEPWATER_IMAGE_PRED_H__
#define __DEEPWATER_IMAGE_PRED_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "include/c_predict_api.h"

class ImagePred {
 public:
  // channel = 3 means we have RGB value; 1 for grey-scale img
  explicit ImagePred(int w = 224, int h = 224, int c = 3);
  ~ImagePred();

  void setSeed(int seed);

  void setModelPath(char * path) {model_path_ = std::string(path);}

  void loadInception();
  void loadModel();

  // return the result with highest prob
  const char * predict(float * data);
  // return probs for each class
  std::vector<float> predict_probs(float * data);

 private:
  // labels for imagenet
  std::vector<std::string> synset;
  std::string model_path_;
  const mx_float * nd_data;

  PredictorHandle pred_hnd;
  NDListHandle nd_hnd;
  int image_size, width, height, channels;

  int dev_type, dev_id;
};

#endif
