#!/usr/bin/env bash

# build tensorflow with gpu
# cd thirdparty
# ./build-tensorflow-gpu.sh

cd ..
./gradlew clean
cp thirdparty/tensorflow/cppbuild-gpu.sh thirdparty/tensorflow/cppbuild.sh
./gradlew tensorflowCompile publishToMavenLocal build -x test
