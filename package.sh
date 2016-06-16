#!/bin/bash

javac *.java

rm -rf water/gpu

mkdir -p water/gpu

mv *.class ./water/gpu

jar -cvf water.gpu.jar ./water
