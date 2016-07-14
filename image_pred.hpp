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
  ImagePred();
  void setModelPath(char * path) {model_path_ = std::string(path);}
  void loadInception();
  const char * predict(float * data);
  ~ImagePred();

 private:
  std::vector<std::string> synset;
  std::string model_path_;
  const mx_float * nd_data;
  PredictorHandle pred_hnd;
  NDListHandle nd_hnd;
  int image_size, width, height, channels;

  int dev_type, dev_id;
};

#endif
