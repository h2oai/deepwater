## Build Requirements

1. A C++ compiler with C++11 support
1. [SWIG](http://www.swig.org/)
1. [OpenCV](http://opencv.org) - optional

#### Build mxnet 
##### Ubuntu

```bash
cd ../thirdparty/mxnet
git submodule update --init --recursive
cp make/config.mk .
### EDIT config.mk - USE_OPENCV=0, USE_CUDA=1, USE_CUDA_PATH=...
make -j8
cd ..
make -j8 # will generate a jar file water.gpu.jar that includes native code
```

##### Macosx

First install [homebrew](http://brew.sh).
```bash
git submodule update --init --recursive --depth 1
brew update
brew tap homebrew/science
cd ../thirdparty/mxnet; cp make/osx.mk ./config.mk; make -j$(sysctl -n hw.ncpu)
cd ..
make pkg # will generate a `jar` file including native code.
```

For other options see the [official mxnet build instructions](http://mxnet.readthedocs.io/en/latest/how_to/build.html).

#### Build mxnet H2O bindings
Once libmxnet.so is built, go back to the top-level directory of the deepwater repo, and follow the directions there.
