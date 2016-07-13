/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __H2O_IMAGENET_H__
#define __H2O_IMAGENET_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "include/c_predict_api.h"

class ImageNative {
 public:
  ImageNative();
  void setModelPath(char * path) {model_path_ = std::string(path);}
  void loadInception();
  const char * predict(float * data);
  ~ImageNative();

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
