## Installation Overview

This section provides a high-level description of the installation process. 

The image below is a conceptual stack of software that is installed and required for deep water. 

![Installation Stack](images/installation_stack.jpg)

- Setup and Utilities. At the beginning of installation steps, recommended folders will be made for organizational purposes. A clean update of packages is made and a system reboot to apply those updates. Then, utilities for the installation process are installed.- Java. Java is required for H2O, so if the system does not contain Java, it must be installed.- Low Level Libraries. A variety of low-level libraries are installed that are required for deep learning and are dependencies for the backends.- GPU Drivers. To enable, GPU processing, supported GPU drivers must be installed.- CUDA and cuDNN. Once the GPU drivers are installed, we can install CUDA, a general purpose GPU programming API and parallel computing platform from nVIDIA. On top of CUDA, we'll install cuDNN, alibrary of primitives for deep neural networks that leverage the GPU.- C and C++ Compilers. C and C++ compilers are installed for installing required packages as well as for building certain components from source.- Build Tools. A number of tools are then installed for the build process, including: Maven (TensorFlow), Bazel (TensorFlow), SWIG (MXNet), and Node.js (H2O). SWIG is used for H2O to interface with MXNet. Node.js is required for building H2O Flow. Alternatives is an optional install/conguration to manage various versions of build assets, such as versions of compilers. Environment managers, such as virtualenv, are optional installed for customized and isolated install environments.- Python and R. Python and R clients are installed as well as any development tools required to installed modules and packages.

- Deep Water Build. All the above installation steps are simply pre-requisites to use or build Deep Water. Building Deep Water, involves clone the source code and building the various backends. The output of this step will be a JAR file that will be consumed by H2O core. Also, for each backend, the any client (Python or R) API module or package must be installed.
- H2O Build. H2O is built using the Deep Water JAR. The output Python and R artifacts must alsobe installed.- Amazon Machine Image. If the install is designed for public consumption through an Amazon Machine Image (AMI), a number of extra items are installed for convenience and security. These include an auto-launch script and nginx proxy.

### Installation Paths and Stack

Various installation paths allow for certain scenarios and support different use cases.

1. Limit privilege access

   Separating system-wide and user-specific instllation steps will follow the best practice of limiting administrative previliges.
   2. Customize user-specific installations

   Installing the minimum required artifacts system-wide (with administrative previleges) and leaving the rest for individual user installation also allows for customization of the installation per user.3. Isolate installations   The above also isolates user installations, preventing one user's customization to affect other users. Isolation can further be done through the use of installation environments for the clients. Specifically, for Python the use of environments, such as virtualenv or Anaconda and for R, using personal libraries. As client modules or packages are updated, they can be adopted in different environments to allow for different test cases or validation prior to adoption.
   
   ![Installation Paths](images/installation-paths.jpg)

### S3 Bucket

Installation artifacts are stored on S3 at in the bucket h2o-deepwater. The bucket hierarchy is shown in the image below.

![S3 Bucket](images/s3-bucket.png)

At the top level, there are three namespaces: **internal**, **release**, and **public**. 

The internal namespace is private to H2O.ai and is used to store artifacts that should only be used by internal H2O.ai personnel. This can included experimental/development builds, third-party artifacts typically requiring registration, or just a backup repository for artifacts. Within the internal namespace are other namespaces to help organize artifacts:- cuda: CUDA files and versions.- cudnn: cuDNN files and versions.- h2o: H2O jars, H2O Python wheels, and H2O R package (tar.gz).- mxnet: MXNet Python wheels and R package.- prediction services: Standard alone prediction services. Jetty JAR file.- steam: Steam artifacts- tensorflow: TensorFlow artifacts, such as Bazel (required for build) and Protocol Buffers.The release namespace is public and used to store general Deep Water release artifacts. 

The public namespace is public and is used to store artifacts that might be useful for public consumption, but not necessarily part of an ocial build. They may include datasets, development builds, or other useful artifacts.

### Installation Use Cases

How you progress through the installation can change depending on what your end goal is. The followingsubsections describe various use cases.
#### Multiple Deep Water Installation Per User
This use case is a situation where multiple users can potentially install their own version of Deep Water (andaccompany client packages/modules). In this scenario, a best practice is to install common components thatrequire administrative privileges system wide and then have each user install user-specific components thatdo not require administrative privileges. This use case is the motivation behind system-wide and user-specificinstallation paths that are described in later sections.
#### Single Global Deep Water InstallationThis use case is a sitatuion where there is single Deep Water installation for the entire machine. Separatingsystem-wide and user-specific installation steps is still a good practice; this just means the single user hasadministrative privileges, but selectively uses ``sudo`` to install components. Of course, all installation stepscan just be done with administrative privileges as well.#### H2O AMIThis use case is designed for public consumption with minimal friction. In fact, extra installation steps aredone so that processes are automatically started and exposed so users do not necessarily even need to log in (i.e., ``ssh``) to the machine. A single privileged user is provided.

### Infrastructure

While these instructions were designed for and validated on AWS EC2 GPU instances, they may work for other machines (virtual or otherwise).

#### Amazon Web Services EC2 Instances- GPU Instances  
  - g2.2xlarge Instance     * NVIDIA GRID (GK104 "Kepler") GPU (Graphics Processing Unit), 1,536 CUDA cores and 4 GB of video (frame buffer) RAM
     * Intel Sandy Bridge processor running at 2.6 GHz with Turbo Boost enabled, 8 vCPUs (Virtual CPUs)
     * 15 GiB of RAM
     * 60 GB of SSD storage  - p2.8xlarge Instance	  
	  * 8 NVIDIA Tesla K80 Accelerators, each running a pair of NVIDIA GK210 GPUs.

##### Provisioning an EC2 Instance

1. Choose a base AMI. For example:

	![Base AMI](images/base-ami.jpg)

	* Ubunto Server 16.04 LTS (HVM) AMI
	  
	  * US East 1 (N. Virginia) Region AMI ID: ami-e13739f6
	  * US West 2 (Oregon) Region AMI ID: ami-b7a114d7

2. Choose an instance type:

   ![Instance Type](images/instance-type.jpg)

3. Add storage:

   ![Add Storage](images/add-storage.jpg)
   
4. Configure Security Group:

   ![Configure Security Group](images/configure-security-group.jpg)
   
   For example:
   
   * Refer existing group (N. Virginia region): deepwater-v4   * SSH, TCP, 22: SSH into instance   * HTTP, TCP, 80: Browser access (Jupyter notebook)   * HTTPS, TCP, 443: Secure browser access (Jupyter notebook)   * Custom TCP Rule, TCP, 8080: Prediction Builder Service   * Custom TCP Rule, TCP, 9000: Steam   * Custom TCP Rule, TCP, 50325-50535: Steam Deployment Services   * Custom TCP Rule, TCP, 54321-54330: H2O   * Custom TCP Rule, TCP, 55001: Pre-built prediction service (Cat/Dog/Mouse)   * Custom TCP Rule, TCP, 55011: Pre-built prediction service (Inception)   * Custom TCP Rule, TCP, 55021: Pre-built prediction service (Resnet)

5. Add tags:

   ![Add Tags](images/add-tags.jpg)

6. Review the instance and then launch.

   ![Review and Launch](images/review-and-launch.jpg)

#### GPUs
Most Deep Learning frameworks can work with CPUs only. However, in order to take advantage of GPUs with CUDA, please ensure the GPUs are CUDA-enabled CUDA. (Refer to [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).You can find your GPUs on your machine: 

	$ lspci -nn | grep -i VGA -A 12	00:02.0 VGA compatible controller [0300]: Cirrus Logic GD 5446 [1013:00b8]	00:03.0 VGA compatible controller [0300]: NVIDIA Corporation GK104GL [GRID K520] [10de:118a] (rev a1)	00:1f.0 Unassigned class [ff80]: XenSource, Inc. Xen Platform Device [5853:0001] (rev 01)

### Files and Versions

* CUDA: cuda 8.0.44 linux.run* cuDNN: cudnn-8.0-linux-x64-v5.1.tgz* MXNet  - Python: mxnet-0.7.0-py2.7.egg* TensorFlow  - Bazel: bazel-0.3.1-installer-linux-x86 64.sh