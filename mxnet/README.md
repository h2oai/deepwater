## Build Requirements

1. A C++ compiler with C++11 support
1. [SWIG](http://www.swig.org/)
1. [OpenCV](http://opencv.org) - optional

Note: To install a C++ compiler on Mac OS X, you need to download and install [Command Line Tools for Xcode](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/).  On Mac OS X, both SWIG and OpenCV can be installed via [homebrew](http://brew.sh) (see below). 

### Build MXNet 

#### Ubuntu

If you first need to install SWIG and OpenCV that can also be done using apt-get:

```
sudo apt-get update
sudo apt-get install swig
sudo apt-get install libopencv-dev python-opencv
```

Install MXNet: 

```bash
cd thirdparty/mxnet
git submodule update --init --recursive
cp make/config.mk .
### EDIT config.mk - USE_OPENCV=0, USE_CUDA=1, USE_CUDA_PATH=...
make -j8
```

#### Mac OS X

First install [homebrew](http://brew.sh).  If you first need to install SWIG and OpenCV that can also be done using homebrew:

```bash
brew install swig
brew install opencv
```

Install MXNet: 

```bash
git submodule update --init --recursive
brew update
brew uninstall homebrew/science  #just to make sure you have the most recent 
## sudo chown -R $(whoami):admin /usr/local #if you get write permission issues
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

For other options see the [official MXNet build instructions](http://mxnet.io/get_started/ubuntu_setup.html).

### Build and install MXNet Python bindings
To build the Python egg (which can be installed with `easy_install dist/*.egg`), do the following:

```
cd thirdparty/mxnet/python
python setup.py install
```

Now, you'll have the MXNet python module available for creating your own Deep Learning models from scratch.

### Build MXNet H2O bindings
Once `libmxnet.so` is built, go back to the top-level directory of the deepwater repo, and follow the directions there.
