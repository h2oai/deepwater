/*!
 * Copyright (c) 2016 by Contributors
 */
#ifndef __H2O_NETWORK_DEF_H__
#define __H2O_NETWORK_DEF_H__
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "MxNetCpp.h"

static const std::string PoolingPoolTypeValues[] = {
  "avg",
  "max",
  "sum"
};

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
                                 const std::string & suffix = "",
                                 mx_float eps = 0.001,
                                 mx_float momentum = 0.9);

mxnet::cpp::Symbol ConvFactoryNoBias(mxnet::cpp::Symbol data, int num_filter,
                                     mxnet::cpp::Shape kernel,
                                     mxnet::cpp::Shape stride = mxnet::cpp::Shape(1, 1),
                                     mxnet::cpp::Shape pad = mxnet::cpp::Shape(0, 0),
                                     const std::string & name = "",
                                     const std::string & suffix = "");

mxnet::cpp::Symbol InceptionFactoryA(mxnet::cpp::Symbol data, int num_1x1, int num_3x3red,
                                     int num_3x3, int num_d3x3red, int num_d3x3,
                                     mxnet::cpp::PoolingPoolType pool, int proj,
                                     const std::string & name,
                                     mx_float eps = 0.001,
                                     mx_float momentum = 0.9);

mxnet::cpp::Symbol InceptionFactoryB(mxnet::cpp::Symbol data, int num_3x3red, int num_3x3,
                                     int num_d3x3red, int num_d3x3, const std::string & name,
                                     mx_float eps = 0.001,
                                     mx_float momentum = 0.9);

mxnet::cpp::Symbol InceptionSymbol(int num_classes);

mxnet::cpp::Symbol InceptionSymbol2(int num_classes);

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
/* Inception V3 */

mxnet::cpp::Symbol Inception7A(mxnet::cpp::Symbol data,
                               int num_1x1, int num_3x3_red, int num_3x3_1,
                               int num_3x3_2, int num_5x5_red, int num_5x5,
                               mxnet::cpp::PoolingPoolType pool, int proj,
                               const std::string & name);

// First Downsample
mxnet::cpp::Symbol Inception7B(mxnet::cpp::Symbol data,
                               int num_3x3, int num_d3x3_red,
                               int num_d3x3_1, int num_d3x3_2,
                               mxnet::cpp::PoolingPoolType pool,
                               const std::string & name);

mxnet::cpp::Symbol Inception7C(mxnet::cpp::Symbol data,
                               int num_1x1, int num_d7_red, int num_d7_1,
                               int num_d7_2, int num_q7_red, int num_q7_1,
                               int num_q7_2, int num_q7_3, int num_q7_4,
                               mxnet::cpp::PoolingPoolType pool,
                               int proj, const std::string & name);

mxnet::cpp::Symbol Inception7D(mxnet::cpp::Symbol data,
                               int num_3x3_red, int num_3x3, int num_d7_3x3_red,
                               int num_d7_1, int num_d7_2, int num_d7_3x3,
                               mxnet::cpp::PoolingPoolType pool,
                               const std::string & name);

mxnet::cpp::Symbol Inception7E(mxnet::cpp::Symbol data,
                               int num_1x1, int num_d3_red, int num_d3_1,
                               int num_d3_2, int num_3x3_d3_red, int num_3x3,
                               int num_3x3_d3_1, int num_3x3_d3_2,
                               mxnet::cpp::PoolingPoolType pool,
                               int proj, const std::string & name);

mxnet::cpp::Symbol InceptionV3Symbol(int num_classes);

mxnet::cpp::Symbol ConvModule(const std::string & name,
                              mxnet::cpp::Symbol net,
                              mxnet::cpp::Shape kernel_size,
                              mxnet::cpp::Shape pad_size,
                              int filter_count,
                              mxnet::cpp::Shape stride = mxnet::cpp::Shape(1, 1),
                              int work_space = 2048,
                              bool batch_norm = true,
                              bool down_pool = false,
                              bool up_pool = false,
                              const std::string & act_type = "relu",
                              bool convolution = true);

mxnet::cpp::Symbol UNetSymbol();

mxnet::cpp::Symbol MLPSymbol(const std::vector<int> &layerSize, const std::vector<std::string> &activations, int num_classes);
#endif
