# Deep Water

Native implementation of Deep Learning models for GPU backends (mxnet, Caffe, TensorFlow, etc.)

![architecture](./architecture/overview.png "Deep Water High-Level Architecture")

## Requirements

* SWIG: http://www.swig.org/

* A C++ compiler with C++11 support

`make pkg` will generate a `jar` file including native code.

Please add these lines below into your Java code when using this jar as external dependency.

```Java
util.loadCudaLib();
util.loadNativeLib("mxnet");
util.loadNativeLib("Native");
```

An example implementation can be found in [H2O](https://github.com/h2oai/h2o-3/blob/deepwater/h2o-algos/src/test/java/hex/deeplearning/DeepWaterTest.java).
