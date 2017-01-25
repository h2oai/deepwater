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
```

##### Macosx

First install [homebrew](http://brew.sh).
```bash
git submodule update --init --recursive
brew update
brew uninstall homebrew/science
## sudo chown -R $(whoami):admin /usr/local ##if you get write permission issues
brew tap homebrew/science
brew install openblas

cd ../thirdparty/mxnet 
cp make/osx.mk ./config.mk
cat << EOF >> ./config.mk
ADD_LDFLAGS = '-L/usr/local/opt/openblas/lib'
ADD_CFLAGS = '-I/usr/local/opt/openblas/include'
EOF
make -j$(sysctl -n hw.ncpu)
```

For other options see the [official mxnet build instructions](http://mxnet.readthedocs.io/en/latest/how_to/build.html).

#### Build and install mxnet Python bindings
To build the Python egg (which can be installed with `easy_install dist/*.egg`), do the following:
```
cd thirdparty/mxnet/python
python setup.py install
```
Now, you'll have the mxnet python module available for creating your own Deep Learning models from scratch.

#### Build and install mxnet R bindings
To build the R module (which can be installed with `R CMD INSTALL mxnet_current_r.tar.gz`), do the following:
```
cd thirdparty/mxnet/
make rpkg
```
You might have to install or upgrade certain dependencies in R first (with `install.packages("package_name")` in R).
Now, you'll have the mxnet R module available for creating your own Deep Learning models from scratch.

#### Build mxnet H2O bindings
Once libmxnet.so is built, go back to the top-level directory of the deepwater repo, and follow the directions there.
