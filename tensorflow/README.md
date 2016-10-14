# Tensorflow Bindings for H2O.ai

This package provides inference bindings for Tensorflow and H2O.ai.

The module directory contains a pretrained set of ready to use models. 


#### Javaccp presets
```
cd thirdparty
git submodule update --init --recursive
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

#### Install Bazel
```
sudo apt-get install bazel ## Ubuntu, see https://www.bazel.io/versions/master/docs/install.html#ubuntu
brew install bazel ## Mac
```


#### Build TF 
```
cd javacpp-presets
./cppbuild.sh install tensorflow
```

#### Install Maven
```
sudo apt install maven
brew install maven
```

#### Build TF Java bindings
```
mvn -T 20 install --projects .,tensorflow
```

#### Build TF H2O bindings
Once TensorFlow is built, go back to the top-level directory of the deepwater repo, and follow the directions there.
