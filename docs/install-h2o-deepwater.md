# Install H2O Deep Water

To install H2O Deep Water, there are two options available:  

- Use provided pre-compiled files 
- Build from source. 

Note that compiling from source will provide the latest or specific version of H2O Deep Water.

## H2O Deep Water: Artificts

You can install H2O Deep Water artifacts using Python or MXNet.

**H2O Python Module**

Install the H2O python module (Wheel file): ``$ sudo pip install h2o*.whl``

**MXNet**

Install MXNet (EGG file): ``$ sudo easy install mxnet-0.7.0-py2.7.egg``

## H2O Deep Water: Build from Source

### Pre-Requisites

#### C and C++ Compiler

The C++ compiler must have C++11 support. 

  - Verify that the version of the existing g++ is less than 5.3: ``$ g++ -v``

    Note that if the version if higher, you will get an unsupported GNU version exception. 
  
  - Select appropriate gcc and g++ version using alternatives:

  		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --config gcc
		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --config g++
		
#### H2o Deep Water Repository

1. Clone the Deep Water repository: ``$ git clone https://github.com/h2oai/deepwater.git``

2. (Optional) If you want to checkout a specific Deep Water hash:

		$ cd ~/deepwater
		$ git checkout <hash>

	**Note**: A known stable repository commit hash is: 989eac97e71bc958394d0bbec22937e1e05d854f
	
### MXNET

1. Update third party dependencies:

		$ cd ~/deepwater
		$ git submodule update --init --recursive

2. Edit config.mk for CUDA support: From the deepwater/thirdparty/mxnet/ directory:

		$ cp make/config.mk .

   Using the editor of your choice, edit portions of the config.mk as follows. Please make note of what is not commented (i.e. the lines without a # sign):

   		# whether use CUDA during compile
   		USE_CUDA = 1

   		# add the path to CUDA library to link and compile flag
		# if you have already add them to environment variable, leave it as NONE
		USE_CUDA_PATH = /usr/local/cuda
		
		# USE_CUDA_PATH = NONE
		# whether use CuDNN R3 library
		USE_CUDNN = 1
		
		# whether use cuda runtime compiling for writing kernels in native language (i.e. Python)
		USE_NVRTC = 0
		
		# whether use opencv during compilation
		# you can disable it, however, you will not able to use
		# imbin iterator
		USE_OPENCV = 0

3. Compile:

		$ make -j8
		
4. Verify that libmxnet.so is created at /deepwater/thirdparty/mxnet/lib

### MXNet Python Module

The MXNet build process will create a new Python setup file. Install MXNet Python module using the following commands:
	
		$ cd ~/deepwater/thirdparty/mxnet/python
		$ python setup.py install

### MXNet R Package

1. Install Shiny (from R): 

		> install.packages("shiny")

2. Downgrade the version of DiagrammeR. (Refer to [http://stackoverflow.com/questions/41469083/install-error-on-mxnet](http://stackoverflow.com/questions/41469083/install-error-on-mxnet).)

		> library("devtools")
		> library("DiagrammeR")
		> install_version("DiagrammeR", version = "0.8.1", repos = "http://cran.us.r-project.org")

		$ cd ~/deepwater/thirdparty/mxnet/R-package
		$ Rscript -e "library(devtools); library(methods); \
		options(repos=c(CRAN=`https://cran.rstudio.com')); \
		install_deps(dependencies = TRUE)"
		$ cd ..
		$ make rpkg
		$ R CMD INSTALL mxnet_0.7.tar.gz

3. Verify the installation in R: 

		> library("mxnet")

### TensorFlow

1. Get version 3.0 protocol buffers. This assumes you will download and place the file to the protobuf directory made during setup.
	
		$ wget https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86 64.zip
		
2. Update path in .profile: 

		export PATH="$PATH:$HOME/protobuf/bin"

3. Build TensorFlow:

		$ cd ~/deepwater/thirdparty/javacpp/
		$ mvn install
		$ cd ~/deepwater/thirdparty/javacpp-presets/
		$ mvn -T 20 install --projects .,tensorflow
		
### TensorFlow Python Module

The TensorFlow Python module is required to create arbitrary graphs using the Python API. The module depends on the version of TensorFlow and CUDA. There may be a wheel file that you can simply install. (See [https://www.tensorflow.org/get_started/os_setup](https://www.tensorflow.org/get_started/os_setup)). Otherwise, you will need to build a Python module for the specific version of TensorFlow. The following are the instructions to build the Python module for TensorFlow 0.10 with CUDA 8. Please update the instructions accordingly with you specific versions.

1. Clone the TensorFlow repository: ``$ git clone https://github.com/tensorflow/tensorflow``

2. Checkout the desired version: ``$ git checkout r0.10``
3. Navigate to TensorFlow directory: ``$ cd /tensorflow``
4. Configure Bazel:

		$ ./
		Please specify the location of python. [Default is /usr/bin/python]:
		Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] N
		No Google Cloud Platform support will be enabled for TensorFlow
		Do you wish to build TensorFlow with GPU support? [y/N] y
		GPU support will be enabled for TensorFlow
		Please specify which gcc nvcc should use as the host compiler. [Default is /usr/bin/gcc]:
		Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 8.0
		Please specify the location where CUDA 8.0 toolkit is installed.
		Refer to README.md for more details. [Default is /usr/local/cuda]:
		Please specify the cuDNN version you want to use. [Leave empty to use system default]: 5
		Please specify the location where cuDNN 5 library is installed.
		Refer to README.md for more details. [Default is /usr/local/cuda]:
		Please specify a list of comma-separated Cuda compute capabilities you want to build with. You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
		Please note that each additional compute capability significantly increases your build time and binary size. [Default is: "3.5,5.2"]: 3.0
		Setting up Cuda include
		Setting up Cuda lib
		Setting up Cuda bin
		Setting up Cuda nvvm
		Setting up CUPTI include
		Setting up CUPTI lib64
		Configuration finished

5. Update crosstool and add the line below so that the build process can find the CUDA 8 header files:

		$ cd ~/tensorflow/third_party/gpus/crosstool/
		$ vi CROSSTOOL.tpl
		
		cxx builtin include directory: ``/usr/local/cuda-8.0/include''

6. Build with GPU support:

		$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
		$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
		
7. Install the Python module. If you are using a particular environment (virtual environment), please be sure you are in that environment. The exact name of the wheel file will depend on your platform.

		$ pip install /tmp/tensorflow_pkg/tensorflow-0.12.1-py2-none-any.whl

If you encounter any issues, please check the following page for updates or discrepancies: [https://www.tensorflow.org/get_started/os_setup](https://www.tensorflow.org/get_started/os_setup).

### Build Deep Water Jar File

1. Remove TensorFlow from the build process: Modify the following files using your editor of choice.

   - ``settings.gradle``: Comment out line 9: ``// include 'tensorflow'``
   - ``build.gradle``: Comment out line 98: ``// compile project('deepwater-tensorflow')``

2. Build Deep Water:

		$ cd ~/deepwater/
		$ ./gradlew build -x test

3. Verify jar creation: 

		build/libs/deepwater-all.jar
		
### H2O with Deep Water

1. Clone the H2O repository: 

		$ git clone https://github.com/h2oai/h2o-3.git

2. (Optional) Run the folloiwng if you want to checkout a specific H2O hash:

		$ cd ~/h2o-3
		$ git checkout <hash>

	**Note**: A known stable repository commit hash is: 08b86a7f36bbbed478eb2d1c079297f2f74a5dea

3. Link the freshly created Deep Water jar file from the previous section to h2o-3 library. Note that if **~/h2o-3/lib** does not exist, then you must make the directory: ``$ mkdir lib``

		$ cd ~/h2o-3/lib
		$ ln -sf ~/deepwater/build/libs/deepwater-all.jar .

4. Build H2O-3:

		$ cd ~/h2o-3
		$ ./gradlew build -x test

### H2O Python Module
The H2O build process will create a new Python wheel file. Install H2O with the Deep Water Python module:

		$ pip install ~/h2o-3/h2o-py/dist/h2o-*.whl

### H2O R Package

The H2O build process will create a new R package. Install H2O with Deep Water R module:

		$ R CMD INSTALL ~/h2o-3/h2o-r/R/src/contrib/h2o-*.tar.gz

