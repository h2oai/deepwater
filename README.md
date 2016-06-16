# h2o-native

Native implmentation of deep learning model

## Requirement

* SWIG: http://www.swig.org/

* A C++ compiler with C++11 support

On OSX, you might need to run something similar to the line below:

```
install_name_tool -change lib/libmxnet.so "@loader_path/libmxnet.so" libmlp.dylib
```
