# DeepWater

Native implementation of Deep Learning models for GPU backends (mxnet, Caffe, TensorFlow, etc.)

## Requirement

* SWIG: http://www.swig.org/

* A C++ compiler with C++11 support

On OSX, you might need to run something similar to the line below:

```
install_name_tool -change lib/libmxnet.so "@loader_path/libmxnet.so" libNative.dylib
```

`make pkg` will generate a `jar` file including native code.

Please add these lines below into your Java code when using this jar as external dependency.
```
util.loadCudaLib();
util.loadNativeLib("mxnet");
util.loadNativeLib("Native");
```
