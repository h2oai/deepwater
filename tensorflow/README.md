# Tensorflow Bindings for H2O.ai

This package provides inference bindings for Tensorflow and H2O.ai.

The module directory contains a pretrained set of ready to use models. 


#### Javaccp presets
```
cd thirdparty
git submodule update --init --recursive --depth 1
```

#### Install Bazel
```
#sudo apt-get install bazel
#brew install bazel on Mac
```


#### Build TF 
```
cd javacpp-presets
./cppbuild.sh install tensorflow
```

#### Install Maven
```
#sudo apt install maven
#brew install maven
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
ln -sf build/libs/tensorflow-1.0-SNAPSHOT-sources.jar ~/h2o-3/h2o-algos/
ln -sf build/libs/tensorflow-1.0-SNAPSHOT-javadoc.jar ~/h2o-3/h2o-algos/
ln -sf build/libs/deepwater.backends.tensorflow-1.0-SNAPSHOT.jar ~/h2o-3/h2o-algos/
cd ..
ln -sf build/libs/deepwater.backends-1.0-SNAPSHOT.jar ~/h2o-3/h2o-algos/
```
