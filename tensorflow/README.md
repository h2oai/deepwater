# Tensorflow Bindings for H2O.ai

This package provides inference bindings for Tensorflow and H2O.ai.

The module directory contains a pretrained set of ready to use models. 

#### Install Protobuf 3.0.0
https://github.com/google/protobuf/releases/tag/v3.0.0

#### Install Bazel 0.3.1
For Linux:
```
wget https://github.com/bazelbuild/bazel/releases/download/0.3.1/bazel-0.3.1-installer-linux-x86_64.sh
sudo bash bazel-0.3.1-installer-linux-x86_64.sh
exit
```

For Mac:
```
curl -L -O https://github.com/bazelbuild/bazel/releases/download/0.3.1/bazel-0.3.1-installer-darwin-x86_64.sh
sudo bash bazel-0.3.1-installer-darwin-x86_64.sh
exit
```

#### Javaccp presets
```
cd thirdparty
git submodule update --init --recursive
cd ..
```

## Install Anaconda Python

```
conda create --name deepwater python=2.7
source activate deepwater
# The numpy version is important
conda install numpy=1.10
```
#### Install SWIG
```
sudo apt-get install swig
```


#### Install Maven
```
sudo apt install maven
brew install maven
```

#### Build JavaCPP 
```
cd thirdparty/javacpp
mvn install
cd ../..
```

#### Build TF Java bindings
```
cd thirdparty/javacpp-presets
mvn -T 20 install --projects .,tensorflow
cd ../..
```

#### Build TF H2O bindings
Once TensorFlow is built, go back to the top-level directory of the deepwater repo, and follow the directions there.
