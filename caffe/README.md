# Caffe Bindings for H2O.ai

This package provides inference bindings for Caffe and H2O.ai.

The module directory contains a pretrained set of ready to use models.

#### Javaccp presets
```
cd thirdparty
git submodule update --init --recursive
cd ..
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

#### Build Caffe Java bindings
```
cd thirdparty/javacpp-presets
mvn -T 20 install --projects .,opencv,caffe
cd ../..
```

#### Build Caffe H2O bindings
Once Caffe is built, go back to the top-level directory of the deepwater repo, and follow the directions there.
