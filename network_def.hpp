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

