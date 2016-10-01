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
``


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
```
cd ../../
./gradlew build -x test
cd tensorflow
../gradlew build -x test
```

#### Link the created .jars to H2O (deepwater branch)
```
cd ..
ROOT=$PWD
H2ODIR=~/h2o-3/

cd $H2ODIR/h2o-algos/

ln -sf $ROOT/tensorflow/build/libs/tensorflow-1.0-SNAPSHOT-sources.jar .
ln -sf $ROOT/tensorflow/build/libs/tensorflow-1.0-SNAPSHOT-javadoc.jar .
ln -sf $ROOT/tensorflow/build/libs/deepwater.backends.tensorflow-1.0-SNAPSHOT.jar .
ln -sf $ROOT/build/libs/deepwater.backends-1.0-SNAPSHOT.jar .
```
