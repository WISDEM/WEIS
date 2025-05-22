Installation
=================

This page provides instructions for installing WEIS and its dependencies.

Pre-requisites
------------------

Before installing WEIS, ensure you have the following prerequisites:

* **Python Environment**: WEIS requires `Python >=3.9` with the `Anaconda` or `Miniforge3` package managers
* **Operating Systems**: WEIS is supported on:
   * Linux
   * macOS
   * Windows Subsystem for Linux (WSL)
   * Native Windows
* **Compiler Requirements**:
   * Windows: m2w64-toolchain and libpython
   * Linux/Mac: compilers
* **Git**: Required for downloading the source code
* **Additional Packages**: Some optional features require additional packages:
   * petsc4py (Linux/Mac only)
   * mpi4py 
   * pyoptsparse (for optimization capabilities)

For those working behind company firewalls, you may need to change the conda authentication with:
::

   conda config --set ssl_verify no

Proxy servers can also be set with:
::

   conda config --set proxy_servers.http http://id:pw@address:port
   conda config --set proxy_servers.https https://id:pw@address:port


Installation Instructions
-----------------------------

WEIS can be installed following the instructions below. The installation process differs slightly depending on your system.

Standard Installation (Linux, macOS, Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Setup and activate the Anaconda environment:
   ::

      conda config --add channels conda-forge
      conda install git
      git clone https://github.com/WISDEM/WEIS.git
      cd WEIS
      git checkout branch_name                         # (Only if you want to switch branches, say "develop")
      conda env create --name weis-env -f environment.yml
      conda activate weis-env                          # (if this does not work, try source activate weis-env)

2. Add final packages and install the software:
   ::

      conda install -y petsc4py=3.22.2 mpi4py pyoptsparse     # (Mac / Linux only, sometimes Windows users may need to install mpi4py)
      pip install -e .

3. If you want to use the more advanced meshing capability for BEM modeling, install the following after you install WEIS:
   ::

      pip install pygmsh==7.1.17 
      pip install https://github.com/LHEEA/meshmagick/archive/refs/tags/3.4.zip
      pip install trimesh                              # (Only if you want to use the internal function to plot the mesh)

Installation on Kestrel (DOE HPC System)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Purge existing modules and load conda:
   ::

      module purge
      module load conda

2. Setup and activate the Anaconda environment:
   ::

      conda config --add channels conda-forge
      conda install git
      git clone https://github.com/WISDEM/WEIS.git
      cd WEIS
      git checkout branch_name                         # (Only if you want to switch branches, say "develop")
      conda env create --name weis-env -f environment.yml
      conda activate weis-env                          # (if this does not work, try source activate weis-env)

3. Load required modules and install:
   ::

      module load comp-intel intel-mpi mkl
      module unload gcc
      pip install --no-deps -e . -v

Using WEIS After Installation
--------------------------------------

To use WEIS after installation is complete, you will always need to activate the conda environment first:
::

   conda activate weis-env   # (or source activate weis-env)

On Kestrel, make sure to reload the necessary modules.

For Windows users, we recommend installing `git` and the `m264` packages in separate environments as some of the libraries appear to conflict such that WISDEM cannot be successfully built from source. The `git` package is best installed in the `base` environment.

Installing SNOPT for use within WEIS
------------------------------------

SNOPT is a commercial optimization solver that can be used with WEIS through the pyOptSparse package. If you wish to use SNOPT, you'll need to:

1. **Purchase SNOPT**: SNOPT is available for purchase from `Stanford Business Software Inc. <http://www.sbsi-sol-optimize.com/asp/sol_snopt.htm>`_. 

2. **Install SNOPT with pyOptSparse**: There are two approaches to install SNOPT:

   **Option 1: Before installing WEIS (Recommended)**
   
   Before running the final ``pip install -e .`` command for WEIS:

   1. Clone pyOptSparse and build it from source:
      ::
      
         git clone https://github.com/mdolab/pyoptsparse.git
         cd pyoptsparse
      
   2. Create a folder called ``pyoptsparse/pyoptsparse/pySNOPT/source`` if it doesn't exist
      
   3. Place the SNOPT source files in that folder:
      - Copy all files from the ``src`` folder in your SNOPT package 
      - Do **not** include ``snopth.f`` file
      
   4. Install pyOptSparse from source:
      ::
      
         pip install -e .
      
   5. Continue with WEIS installation (``pip install -e .``)
   
   **Option 2: After installing WEIS**
   
   If you've already installed WEIS:
   
   1. Uninstall the conda-installed pyOptSparse:
      ::
      
         conda remove --force pyoptsparse
      
   2. Follow the same steps as in Option 1 to install pyOptSparse from source with SNOPT
   

Additional details on installing SNOPT with pyOptSparse can be found in the `pyOptSparse documentation <https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/SNOPT.html>`_.

.. note::
   SNOPT is particularly useful for constrained optimization problems in WEIS. It is not required but provides enhanced capabilities compared to the open-source optimizers that come with WEIS.