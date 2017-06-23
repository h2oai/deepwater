#!/usr/bin/env bash

# build mxnet with CPU

cd mxnet
cat make/config.mk | sed 's/USE_CUDA.*/USE_CUDA = 0/' | \
sed 's/USE_CUDNN.*/USE_CUDNN = 0/' | \
sed 's/USE_CUDA_PATH.*/USE_CUDA_PATH = NONE/' | \
sed 's/USE_OPENCV.*/USE_OPENCV = 0/' > config.mk
make clean
make -j$(nproc)

