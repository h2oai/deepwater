#!/bin/bash

javac *.java

rm -rf water/gpu

mkdir -p water/gpu

mv *.class ./water/gpu

cp ./libNative.so ./water/gpu
cp ./libmxnet.so ./water/gpu

jar -cvf water.gpu.jar ./water
