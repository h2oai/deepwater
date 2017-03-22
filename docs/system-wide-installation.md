## System-Wide Installation Setup and Requirements

### Setup

1. Create a build folder using: ``$ mkdir build``
2. Update **apt**:

		$ sudo apt-get update
		$ sudo apt-get upgrade
		$ sudo apt-get dist-upgrade
		$ sudo reboot

### Utilities

1. Git. Git should be pre-installed with Ubuntu 16.04. You can verify the install with: ``$ git --version.``

	If Git is not found, you can install it: ``$ sudo apt-get install git.``

2. Unzip. Install unzip using: ``$ sudo apt-get install unzip``
		
### Java

We will install Oracle Java 8.

1. Add repository using: ``$ sudo add-apt-repository ppa:webupd8team/java``

2. Update using: ``$ sudo apt-get update``

3. Install using: ``$ sudo apt-get install oracle-java8-installer``

	You must accept the license agreements.

4. Verify installation using: ``$ java -version``

### Low-Level Libraries


1. Install Basic Linear Algebra Subroutines (BLAS) library using: ``$ sudo apt-get install libblas-dev``
2. Install Automatically Tuned Linear Algebra Software (ATLAS) using: ``$ sudo apt-get install libatlas-dev``
3. Install Linear Algebra Routines: ``$ sudo apt-get install liblapack-dev``
4. Install Development Files for OpenCV using: ``$ sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev``
5. Install GCC OpenMP (GOMP) using: ``$ sudo apt-get install libgomp1``

  Note: This should already be installed with Ubuntu 16.04.

### NVIDIA GPU Drivers

At the time of this writing, the latest version of NVIDIA drivers is 370. Instructions reflect this and are verified to work with this version throughout this document. If using a different version, edit the instruction commands to reflect the version that you are using.

1. Clean old install: ``$ sudo apt-get purge nvidia-*``
2. (Optional) Find/verify available drivers

    Without descriptions: ``$ apt-cache search nvidia``
    
    With descriptions: ``$ apt search nvidia``
3. Add repository: ``$ sudo add-apt-repository ppa:graphics-drivers/ppa``
4. Update repository: ``$ sudo apt-get update``
5. Verify NVIDIA 370 drivers available: ``$ apt search nvidia | grep 370``

    You should see a result that starts with nvidia-370/xenial 370.
  
6. Install driver and configuration tool: ``$ sudo apt-get install nvidia-370 nvidia-settings``
7. Reboot the server: ``$ sudo reboot``
8. Test the installation: ``$ nvidia-smi``
9. **IMPORTANT**: Kernel Updates

   From here, DO NOT run a package manager update (i.e. ``$ sudo apt-get update``). Kernel updates will likely cause NVIDIA drivers to no longer function, as they are usually specific to kernel version and will result in a failed communication error.
   
### Monitoring GPU Usage

Run the following to monitor GPU usage:

		watch -d -n 1 nvidia-smi
		
### Disable Automatic System Updates

By default, Ubuntu applies automatic system updates, which as previously indicated, can cause the NVIDIA driver to fail. Consequently, we will stop auto system updates because after every kernel update, NVIDIA driver fails to communicate with system.

		$ sudo apt-get remove unattended-upgrades

### CUDA

At the time of the this writing, Deep Water is developed to support only CUDA 8. Instructions reflect this and are verified to work with this version throughout this document. If you are using a different version, edit the commands to reflect the version that you are using. 

**Note**: Before you begin, we recommend that you save the CUDA files in the **/build** directory created during setup.

1. Get CUDA Files:

	CUDA can be obtained directly  NVIDIA, which requires a registration. You can obtain these files from by going to the NVIDIA web site or using the CLI. 
	
	**Via Web Site**
	
	1. Download the CUDA Toolkit  [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Note that the toolkit is approximately 1.4 GB and may take some time to download. 
	2. Select the following options for the target platform:
		- Operating System: Linux
		- Architecture: x86_64
		- Distribution: Ubuntu
		- Version: 16.04
		- Installer Type: runfile (local)
	
	**Via CLI**
	
	Run the following CLI commands to download CUDA files via the CLI.
	
		$ wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
		$ mv cuda_8.0.44_linux-run cuda_8.0.44_linux.run

2. Install CUDA:

	Because the GPU driver is installed seperately, do not install drivers again through the CUDA installation process. Launch the installation, and answer the prompt questions. Edit to reflect your CUDA version. Note that you may not see all of these questions.
	
	1. Navigate to CUDA fles: ``~/build``
	2. Use a root shell: ``$ sudo bash``
	3. Launch CUDA installation: ``$ bash ./cuda 8.0.44 linux.run -override``

       **Note**: The ``-override`` flag is to avoid an Ubuntu 16.04 "Unsupported compiler error".
      
	4. EULA
		
			Do you accept the previously read EULA?
			accept/decline/quit: accept
	5. Graphics Driver (SELECT NO)

			Install NVIDIA Accelerated Graphics Driver for Linux-x86\_64 361.77?
			(y)es/(n)o/(q)uit: no
	6. OpenGL
		
			Do you want to install the OpenGL libraries?
			(y)es/(n)o/(q)uit [ default is yes ]:
	
	7. nvidia-xconfig
		
			Do you want to run nvidia-xconfig?
			This will update the system X configuration file so that the NVIDIA X driver is used.
			The pre-existing X configuration file will be backed up. This option should not be used on systems that require a custom X configuration, such as systems with multiple GPU vendors.
			(y)es/(n)o/(q)uit [ default is no ]:

	8. CUDA 8.0 Toolkit

			Install the CUDA 8.0 Toolkit?
			(y)es/(n)o/(q)uit: yes

	9. Toolkit Location

			Enter Toolkit Location
			[ default is /usr/local/cuda-8.0 ]:

	10. CUDA Symbolic Link
			
			Do you want to install a symbolic link at /usr/local/cuda?
			(y)es/(n)o/(q)uit: yes

	11. CUDA 8.0 Samples

			Install the CUDA 8.0 Samples?
			(y)es/(n)o/(q)uit: yes
			Enter CUDA Samples Location
			[ default is /home/ubuntu ]:

3. Ignore Warning: You can ignore following warning which is displayed after CUDA installation as we
have separately installed NVIDIA driver.

		**WARNING**: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 361.00 is required for CUDA 8.0 functionality to work. To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file: ``sudo <CudaInstaller>.run -silent -driver``
		
4. Exit root: ``exit``

### cuDNN

**Note**: Before you begin, we recommended that you save the cuDNN files in the /build directory created during setup.

1. Get cuDNN files. cuDNN can be obtained directly  NVidia.

2. Extract cuDNN files: ``$ tar -xvf cudnn-8.0-linux-x64-v5.1.tgz``

   **Note**: This will extract files into a cuda directory.
   
3. Copy cuDNN files to system CUDA directories. We assume you are in the same directory you extracted the cuDNN files.

	Copy lib files: ``$ sudo cp cuda/lib64/* /usr/local/cuda/lib64/``

	Copy include files: ``$ sudo cp cuda/include/* /usr/local/cuda/include/``


### C and C++ Compilers

Throughout the installation process you may need various versions of C and C++ compilers. You can manage which version is used manually, or you can use alternatives, which are described in Alternative section that follows. 

**gcc**

1. Install latest version: ``$ sudo apt-get install gcc``
2. Install version 4.8: ``$ sudo apt-get install gcc-4.8``
3. Check the used version: ``$ gcc -v``

	Note that for Ubuntu 1604, the used (latest) version is 5.4. 
4. Verify all installed versions of gcc: ``$ ls -l /usr/bin/gcc*`` 

   You should see entries for gcc-5 and gcc-4.8 similar to the following. Note that the default of gcc is specified as a symbolic link.
	
		/usr/bin/gcc -> gcc-5
		/usr/bin/gcc-4.8
		/usr/bin/gcc-5

**g++**

1. Install latest version: ``$ sudo apt-get install g++``
2. Install version 4.8: ``$ sudo apt-get install g++-4.8``
3. Check the used version: ``$ g++ -v``

   Note that for Ubuntu 1604, the used (latest) version is 5.4. 

4. Verify all installed versions of g++: ``$ ls -l
/usr/bin/g++*``

   You should see entries for g++-5 and gcc-4.8 similar to the following. Note that the default of g++ is specified as a symbolic link.

		/usr/bin/g++ -> g++-5
		/usr/bin/g++-4.8
		/usr/bin/g++-5

### Maven

1. Install Maven: ``$ sudo apt-get install maven``
2. Verify Maven version: ``$ mvn -v``

### Bazel

Bazel is required to build TensorFlow.

1. Download installer script (recommend into the build directory):

		$ wget https://github.com/bazelbuild/bazel/releases/download/0.3.1/ bazel-0.3.1-installer-linux-x86_64.sh
	
2. Change permission: ``$ chmod +x bazel-0.3.1-installer-linux-x86 64.sh``
3. Install: ``sudo bash bazel-0.3.1-installer-linux-x86 64.sh``
4. Verify the installation and version: ``$ bazel version``

### Python Development Tools

1. Install header files and static library for Python: ``$ sudo apt-get install python-dev``
2. Install Python package manager, pip: ``$ sudo apt-get install python-pip``

### R

Install R by running the following commands:

	$ sudo echo ``deb http://cran.rstudio.com/bin/linux/ubuntu xenial/'' | sudo tee -a /etc/apt/sources.list
	$ gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
	$ gpg -a --export E084DAB9 | sudo apt-key add -
	$ sudo apt-get install r-base r-base-dev

The R development tools (devtools) requires OpenSSL, which can be installed using the following commands:

1. Install OpenSSL flavor and XML library: ``$ sudo apt-get install libcurl4-openssl-dev libxml2-dev``
2. Install SSL toolkit: ``$ sudo apt-get install libssl-dev``

### SWIG

SWIG is required to build H2O and can be installed by running the following command:

	$ sudo apt-get install swig

### Node.js

Node.js is required to build H2O and can be installed by running the following commands:

1. Get repository: ``$ curl -sL https://deb.nodesource.com/setup 4.x | sudo -E bash -``
2. Install node.js: ``$ sudo apt-get install -y nodejs``
3. Verify the installation: ``$ nodejs -v``

### Alternatives

Throughout the installation process you may need various versions of C and C++ compilers. You can manage the version is used manually, or you can use alternatives.

#### Alternatives for C and C++ Compilers

**gcc C compiler**

Run the following commands to use alternatives for the C compiler:

	$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 100
	$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
	$ sudo update-alternatives --config gcc

Verify alternatives using: ``$ ls -l /usr/bin/gcc*``

You should see a response similar to the following: ``/usr/bin/gcc -> /etc/alternatives/gcc``

**g++ C++ compiler**

Run the following commands to use alternatives for the C++ compiler:

	$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 100
	$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50
	$ sudo update-alternatives --config g++

Verify alternatives using: ``$ ls -l /usr/bin/g++*``

You should see a response similar to the following: ``/usr/bin/g++ -> /etc/alternatives/g++``

### Virtual Environments

Install a virtual environment using the following command: ``$sudo apt-get install virtualenv``
