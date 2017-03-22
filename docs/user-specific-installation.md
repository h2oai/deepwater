# User-Specific Installation Setup and Requirements

This section assumes the user's home directory is the default directory and that the system-wide installation has been completed by an admin. 

## Setup

1. Create a local bin directory: ``mkdir bin``

2. Create a local etc for alternatives: ``mkdir -p etc/alternatives``
		
3. Create a local var for alternatives: ``mkdir -p var/lib/dpkg/alternatives``
		
4. Create a directory for protocol buffer files: ``mkdir protobuf``

5. Create a prediction services directory: ``mkdir prediction_services``

6. Update ``path`` to include the new local bin directories.

		vi .profile
		export PATH=''$HOME/bin:$HOME/ .local/bin:$PATH''
		source .profile
		
## Alternatives

Alternatives writes files to /etc and /var. This requires administration privileges, but we provide user-accessible alternatives.

1. gcc C compiler

		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --install ~/bin/gcc gcc /usr/bin/gcc-5 100
		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --install ~/bin/gcc gcc /usr/bin/gcc-4.8 50
		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --config gcc

2. g++ C++ compiler

		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --install ~/bin/g++ g++ /usr/bin/g++-5 100
		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --install ~/bin/g++ g++ /usr/bin/g++-4.8 50
		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --config g++

## CUDA Environment Variables

1. Set CUDA environment variables in ``.profile``:

		export CUDA PATH=/usr/local/cuda
		export LD LIBRARY PATH=$CUDA PATH/lib64:$LD LIBRARY PATH
		export PATH=$PATH:$CUDA PATH/bin

2. Run .profile: ``$ source .profile``

## Python Virtual Environments

1. Create environment: ``$ virtualenv <ENV>``
	
	Example: ``$ virtualenv ~/dw``

2. To activate the environment: ``$ source ~/<ENV>/bin/activate``
	
	Example: ``$ source ~/dw/bin/activate``

	**Note**: Be sure to activate this environment in any new shells.
	
## Python Modules

These are the Python packages required for the H2O Python client. If using virtual environments, please be sure to activate and install packages in environments. 

1. Install required H2O Python modules:

		$ pip install future
		$ pip install requests
		$ pip install grip
		$ pip install tabulate
		$ pip install numpy
		$ pip install colorama

2. Install Python modules required for demos:

		$ pip install pandas
		$ pip install graphviz
		$ pip install matplotlib
		$ pip install scipy
		$ sudo apt-get install python-scipy
		$ sudo apt-get install graphviz

## R Packages

All commands that are to be executed from R will be pre-fixed with a greater than sign (i.e. >).

1. Update gcc and g++ to version > 5 (devtools requirement). Select appropriate version using alternatives:

		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --config gcc
		$ update-alternatives --altdir ~/etc/alternatives --admindir ~/var/lib/dpkg/alternatives --config g++

2. Start R:

		$ R		

3. From R, install R packages required for H2O:

		> install.packages("devtools", dependencies=TRUE)
		Installing package into /usr/local/lib/R/site-library (as lib is unspecified)
		Warning in install.packages("devtools", dependencies = TRUE) :
		'lib = "/usr/local/lib/R/site-library"' is not writable
		Would you like to use a personal library instead? (y/n) y
		Would you like to create a personal library ~/R/x86_64-pc-linux-gnu-library/3.2 to install packages into? (y/n) y
		--- Please select a CRAN mirror for use in this session ---
		HTTPS CRAN mirror
		48: USA (CA 1) [https]
		> install.packages("RCurl")
		> install.packages("statmod")
		> install.packages("jsonlite")
		> install.packages("h2o")
		
## Jupyter Notebooks

If using virtual environments, please be sure to activate  and install packages in environments. 

	$ pip install jupyter

## Jupyter Notebook R Kernel

**For Virtual Environment**

1. Install required packages:

		> install.packages(c("repr", "IRdisplay", "crayon", "pbdZMQ"))
		> devtools::install_github("IRkernel/IRkernel")

2. Get directory where the IRkernel is installed: 

		> system.file(package="IRkernel")

	Make note of the directory. Let's refer to it as "kernel dir". An example of "kernel dir" is: ``/home/wen/R/x85 64-pc-linux-gnu-library/3.2/IRkernel``

3. Get bin directory of R home: 

		> R.home("bin")

	Make note of the directory. Let's refer to it as "r home". An example of "r home" is: ``/usr/lib/R/bin``

4. Exit R: 

		> exit

	**Note**: You may get prompted to specify whether you to save the workspace image. You can select no.
	
5. Go to kernelspec directory in "kernel dir". For example: 

		$ cd /home/wen/R/x86 64-pc-linux-gnu-library/3.2/IRkernel/kernelspec

6. Optional: Make copy of kernel.json file in case of mistake: 

		$ cp kernel.json kernel.json.orig

7. Modify kernel.json file to point to R in "kernel dir". The first item of the ``argv`` list in the kernel.json file will simply have "R". This needs to be updated with your "r home" directory path. Below are the before and after modi cations of kernel.json (only the argv item should be di fferent. 

   - kernel.json Before

			{
			"argv": ["R",
			"--slave", "-e", "IRkernel::main()", "--args", 		"{connection_file}"],
			"display_name":"R", "language":"R"
			}

   - kernel.json After

			{
			"argv": ["/usr/lib/R/bin/R",
			"--slave", "-e", "IRkernel::main()", "--args", 		"{connection_file}"],
			"display_name": "R", "language": "R"
			}

8. Install kernel with jupyter (replace with your path):

		jupyter kernelspec install --replace --name ir --user /home/wen/R/x86_64-pc-linux-gnu-library/3.2/IRkernel/kernelspec
		
   You should get a confirmation message similar to:

		[InstallKernelSpec] Installed kernelspec ir in /home/wen/.local/share/jupyter/kernels/ir
		
**For Global Environment**

		$ R
		> install.packages(c('repr', 'IRdisplay', 'crayon', 'pbdZMQ', 'devtools'))
		> devtools::install_github('IRkernel/IRkernel')
		> IRkernel::installspec()
