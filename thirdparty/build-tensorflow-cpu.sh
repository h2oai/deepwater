#!/usr/bin/env bash

cd ..
./gradlew clean
cp thirdparty/tensorflow/cppbuild-cpu.sh thirdparty/tensorflow/cppbuild.sh
./gradlew tensorflowCompile publishToMavenLocal build -x test
