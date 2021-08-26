# dt-qp-py-project
## Python version and IDE
- I am currently using python 3.9, and spyder as the IDE [https://www.spyder-ide.org/](https://www.spyder-ide.org/).

## Python Dependencies
- Numpy, Scipy, Matplotlib,pyoptsparse, and OSQP are the current packages that are being used
- mat4py needed to load matfiles in fowt example


## To install OSQP, follow the instructions on the following link
### Windows installation
- [https://osqp.org/docs/get_started/python.html](https://osqp.org/docs/get_started/python.html)
- You will need to install GCC [https://gcc.gnu.org/](https://jmeubank.github.io/tdm-gcc/articles/2021-05/10.3.0-release), Cmake [https://cmake.org/](https://cmake.org/), and if you are using windows, you will need visual studio as well [https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017)
- After these programs are available, build osqp using the instructions given at the end of the page [https://osqp.org/docs/get_started/python.html](https://osqp.org/docs/get_started/python.html)
 
 ### Installing osqp in a linux environment
 Installing osqp for a linux environment is more tedious than installing it for windows
 - First you will need [GCC](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/),
 - Cmake requires GCC and [openSSL](https://www.openssl.org/) to intall
 - After installing GCC and cmake, use the directions in [osqp-python](https://osqp.org/docs/get_started/python.html) guide to install osqp

## Installation instructions for pyoptsparse
### Linux installation
- Currently pyoptsparse has support only in linux
- Follow the installation instructions in [pyoptsparse installation page](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/install.html)
- Building pyoptsparse through conda is straightforward, and I used that method to build pyoptsparse
