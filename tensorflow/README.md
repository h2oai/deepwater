# TensorFlow Bindings for H2O.ai

This package provides inference bindings for TensorFlow and H2O.ai.

The module directory contains a pretrained set of ready-to-use models. 

#### Install Protobuf 3.0.0
[https://github.com/google/protobuf/releases/tag/v3.0.0](https://github.com/google/protobuf/releases/tag/v3.0.0)

#### Install Bazel 0.3.1
For Linux:
```
wget https://github.com/bazelbuild/bazel/releases/download/0.3.1/bazel-0.3.1-installer-linux-x86_64.sh
sudo bash bazel-0.3.1-installer-linux-x86_64.sh
exit
```

For Mac:
(Optional) Mac OS X Sierra needs a Bazel >=0.4 version due to bugs in 0.3.1. It can be installed with `brew install bazel`

```
curl -L -O https://github.com/bazelbuild/bazel/releases/download/0.3.1/bazel-0.3.1-installer-darwin-x86_64.sh
sudo bash bazel-0.3.1-installer-darwin-x86_64.sh
exit
```

#### Javacpp presets
```
git submodule update --init --recursive
```

#### Install Anaconda Python

```
conda create --name deepwater python=2.7
source activate deepwater
conda install numpy=1.10 # The numpy version is important
```
#### Install SWIG
For Linux:
```
sudo apt-get install swig
```

For Mac:
```
brew install swig
```

#### Install Maven
For Linux:
```
sudo apt install maven
```

For Mac:
```
brew install maven
```

#### Build JavaCPP 
```
cd thirdparty/javacpp
mvn install
cd ../..
```

#### Build TF Java bindings
(Optional) To build bindings for GPU disabled TF, comment out the `export TF_NEED_CUDA=1` line and remove `-conf=cuda` in `thirdparty/javacpp-presets/tensorflow/cppbuild.sh` for your platform.

```
cd thirdparty/javacpp-presets
mvn -T 20 install --projects .,tensorflow
cd ../..
```

#### Build TF H2O bindings
Once TensorFlow is built, go back to the top-level directory of the deepwater repo, and follow the directions there.
