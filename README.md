# Deep Water

### What it is
* Native implementation of Deep Learning models for GPU-optimized backends (mxnet, Caffe, TensorFlow, etc.)
* State-of-the-art Deep Learning models trained from the H2O Platform
* The next best thing after sliced bread
* Under development

### What it is not
* An oil drilling platform

### Roadmap, Architecture and Demo
Download the [Deep Water overview slides](./architecture/deepwater_overview.pdf).

![](./architecture/deepwater_overview/deepwater_overview.001.jpeg "Deep Water Roadmap")
![architecture](./architecture/deepwater_overview/deepwater_overview.002.jpeg "More Data")
![architecture](./architecture/deepwater_overview/deepwater_overview.003.jpeg "Deep Water Networks")
![architecture](./architecture/deepwater_overview/deepwater_overview.004.jpeg "Deep Water Architecture")
![architecture](./architecture/deepwater_overview/deepwater_overview.005.jpeg "Deep Water Example in Flow")

### Build instructions here:
If you want to use Deep Water in H2O-3, you'll need to have a .jar file that includes backend support for at least one of MXNet, Caffe or TensorFlow.

#### Build mxnet 
[Instructions to build MXNet](https://github.com/h2oai/deepwater/tree/master/mxnet)

#### Build TensorFlow 
[Instructions to build TensorFlow](https://github.com/h2oai/deepwater/tree/master/tensorflow)

#### Build Caffe 
Coming soon.

#### Build H2O Backend Connectors
From the top-level of the deepwater repository, do
```
./gradlew build -x test
```

This will create a file that matches this pattern: `build/libs/*-all.jar`

### H2O-3 has DeepWater support
You need to check out the 'deepwater' branch of github.com/h2oai/h2o-3.

### Java example use cases
Example [Java GPU-enabled unit tests](https://github.com/h2oai/h2o-3/tree/deepwater/h2o-algos/src/test/java/hex/deepwater).

### Python example use cases
Example [Python GPU-enabled unit tests](https://github.com/h2oai/h2o-3/tree/deepwater/h2o-py/tests/testdir_algos/deepwater).

### R example use cases
Coming soon.

### Scala / Sparkling Water example use cases
Coming soon.
