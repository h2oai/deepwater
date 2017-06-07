#!/usr/bin/env bash

# build mxnet with gpu

cd mxnet
cat make/config.mk | sed 's/USE_CUDA = 0.*/USE_CUDA = 1/' | \
sed 's/USE_CUDNN.*/USE_CUDNN = 1/' | \
sed 's/USE_CUDA_PATH.*/USE_CUDA_PATH = \/usr\/local\/cuda\//' | \
sed 's/USE_OPENCV.*/USE_OPENCV = 0/' > config.mk
make clean
make -j$(nproc)

