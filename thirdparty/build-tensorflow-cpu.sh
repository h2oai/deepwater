#!/usr/bin/env bash

# build tensorflow with cpu
# cd thirdparty
# ./build-tensorflow-cpu.sh

cd ..
./gradlew clean
cp thirdparty/tensorflow/cppbuild-cpu.sh thirdparty/tensorflow/cppbuild.sh
./gradlew tensorflowCompile publishToMavenLocal build -x test
