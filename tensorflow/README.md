# Tensorflow Bindings for H2O.ai

This package provides inference bindings for Tensorflow and H2O.ai.

The module directory contains a pretrained set of ready to use models. 


## Javaccp presets
cd thirdparty
git submodule update --init --recursive

## Install Bazel
#sudo apt-get install bazel
#brew install bazel on Mac

## Build TF 
cd javacpp-presets
./cppbuild.sh install tensorflow 

## Install Maven
#sudo apt install maven

## Build TF Java bindings
mvn -T 20 install --projects .,tensorflow

## Build TF H2O bindings
cd ../../
./gradlew build -x test
cd tensorflow
../gradlew build -x test

