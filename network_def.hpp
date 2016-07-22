/*!
 * Copyright (c) 2016 by Contributors
 */
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "MxNetCpp.h"

mxnet::cpp::Symbol AlexnetSymbol(int num_classes);

mxnet::cpp::Symbol InceptionFactory(mxnet::cpp::Symbol data, int num_1x1, int num_3x3red,
                                    int num_3x3, int num_d5x5red, int num_d5x5,
                                    mxnet::cpp::PoolingPoolType pool, int proj,
                                    const std::string & name);

mxnet::cpp::Symbol GoogleNetSymbol(int num_classes);

mxnet::cpp::Symbol ConvFactory(mxnet::cpp::Symbol data, int num_filter,
                               mxnet::cpp::Shape kernel,
                               mxnet::cpp::Shape stride = mxnet::cpp::Shape(1, 1),
                               mxnet::cpp::Shape pad = mxnet::cpp::Shape(0, 0),
                               const std::string & name = "",
                               const std::string & suffix = "");

mxnet::cpp::Symbol ConvFactoryBN(mxnet::cpp::Symbol data, int num_filter,
                                 mxnet::cpp::Shape kernel,
                                 mxnet::cpp::Shape stride = mxnet::cpp::Shape(1, 1),
                                 mxnet::cpp::Shape pad = mxnet::cpp::Shape(0, 0),
                                 const std::string & name = "",
                                 const std::string & suffix = "");

mxnet::cpp::Symbol InceptionFactoryA(mxnet::cpp::Symbol data, int num_1x1, int num_3x3red,
                                     int num_3x3, int num_d3x3red, int num_d3x3,
                                     mxnet::cpp::PoolingPoolType pool, int proj,
                                     const std::string & name);

mxnet::cpp::Symbol InceptionFactoryB(mxnet::cpp::Symbol data, int num_3x3red, int num_3x3,
                                     int num_d3x3red, int num_d3x3, const std::string & name);

mxnet::cpp::Symbol InceptionSymbol(int num_classes);

mxnet::cpp::Symbol VGGSymbol(int num_classes);

mxnet::cpp::Symbol LenetSymbol(int num_classes);

mxnet::cpp::Symbol getConv(const std::string & name, mxnet::cpp::Symbol data,
                           int  num_filter,
                           mxnet::cpp::Shape kernel,
                           mxnet::cpp::Shape stride,
                           mxnet::cpp::Shape pad,
                           bool with_relu,
                           mx_float bn_momentum);

mxnet::cpp::Symbol makeBlock(const std::string & name,
                             mxnet::cpp::Symbol data, int num_filter,
                             bool dim_match, mx_float bn_momentum);

mxnet::cpp::Symbol getBody(mxnet::cpp::Symbol data, int num_level,
                           int num_block, int num_filter, mx_float bn_momentum);

mxnet::cpp::Symbol ResNetSymbol(int num_class, int num_level = 3, int num_block = 9,
                                int num_filter = 16, mx_float bn_momentum = 0.9,
                                mxnet::cpp::Shape pool_kernel = mxnet::cpp::Shape(8, 8));

class LSTMState {
 public:
  LSTMState(mxnet::cpp::Symbol c,
            mxnet::cpp::Symbol h): c(c), h(h) {}
  mxnet::cpp::Symbol c;
  mxnet::cpp::Symbol h;
};

class LSTMParam {
 public:
  LSTMParam(mxnet::cpp::Symbol i2hWeight,
            mxnet::cpp::Symbol i2hBias,
            mxnet::cpp::Symbol h2hWeight,
            mxnet::cpp::Symbol h2hBias):i2hWeight(i2hWeight),
    i2hBias(i2hBias), h2hWeight(h2hWeight), h2hBias(h2hBias) {}
  mxnet::cpp::Symbol i2hWeight;
  mxnet::cpp::Symbol i2hBias;
  mxnet::cpp::Symbol h2hWeight;
  mxnet::cpp::Symbol h2hBias;
};

LSTMState lstm(int num_hidden, mxnet::cpp::Symbol indata,
               LSTMState prev_state, LSTMParam param,
               int seqidx, int layeridx,
               mx_float dropout = 0.0);

mxnet::cpp::Symbol lstm_unroll(int num_lstm_layer, int seq_len,
                               int input_size, int num_hidden,
                               int num_embed, int num_label,
                               mx_float dropout = 0.0);
