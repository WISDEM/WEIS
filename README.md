# WEIS

[![Coverage Status](https://coveralls.io/repos/github/WISDEM/WEIS/badge.svg?branch=develop)](https://coveralls.io/github/WISDEM/WEIS?branch=develop)
[![Actions Status](https://github.com/WISDEM/WEIS/workflows/CI_WEIS/badge.svg?branch=develop)](https://github.com/WISDEM/WEIS/actions)
[![Documentation Status](https://readthedocs.org/projects/weis/badge/?version=develop)](https://weis.readthedocs.io/en/develop/?badge=develop)
[![DOI](https://zenodo.org/badge/289320573.svg)](https://zenodo.org/badge/latestdoi/289320573)

WEIS, Wind Energy with Integrated Servo-control, performs multifidelity co-design of wind turbines. WEIS is a framework that combines multiple NREL-developed tools to enable design optimization of floating offshore wind turbines.

Author: [NREL WISDEM & OpenFAST & Control Teams](mailto:systems.engineering@nrel.gov)

## Documentation

See local documentation in the `docs`-directory or access the online version at <https://weis.readthedocs.io/en/latest/>

## Packages

WEIS integrates in a unique workflow four models:
* [WISDEM](https://github.com/WISDEM/WISDEM) is a set of models for assessing overall wind plant cost of energy (COE).
* [OpenFAST](https://github.com/OpenFAST/openfast) is the community model for wind turbine simulation to be developed and used by research laboratories, academia, and industry.
* [TurbSim](https://www.nrel.gov/docs/fy09osti/46198.pdf) is a stochastic, full-field, turbulent-wind simulator.
* [ROSCO](https://github.com/NREL/ROSCO) provides an open, modular and fully adaptable baseline wind turbine controller to the scientific community.

In addition, three external libraries are added:
* [pCrunch](https://github.com/NREL/pCrunch) is a collection of tools to ease the process of parsing large amounts of OpenFAST output data and conduct loads analysis.
* [pyOptSparse](https://github.com/mdolab/pyoptsparse) is a framework for formulating and efficiently solving nonlinear constrained optimization problems.

The core WEIS modules are:
 * _aeroelasticse_ is a wrapper to call [OpenFAST](https://github.com/OpenFAST/openfast)
 * _control_ contains the routines calling [ROSCO](https://github.com/NREL/ROSCO) and the routines supporting distributed aerodynamic control devices, such trailing edge flaps
 * _gluecode_ contains the scripts glueing together all models and libraries
 * _multifidelity_ contains the codes to run multifidelity design optimizations
 * _optimization_drivers_ contains various optimization drivers
 * _schema_ contains the YAML files and corresponding schemas representing the input files to WEIS

## Installation

On laptop and personal computers, installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WEIS requires [Anaconda 64-bit](https://www.anaconda.com/distribution/). However, the `conda` command has begun to show its age and we now recommend the one-for-one replacement with the [Miniforge3 distribution](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3), which is much more lightweight and more easily solves for the package dependencies.  WEIS is currently supported on Linux, Mac, and Windows (natively or with the Linux sub-system (WSL)).

The installation instructions below use the environment name, "weis-env," but any name is acceptable. For those working behind company firewalls, you may have to change the conda authentication with `conda config --set ssl_verify no`.  Proxy servers can also be set with `conda config --set proxy_servers.http http://id:pw@address:port` and `conda config --set proxy_servers.https https://id:pw@address:port`.

0.  On the DOE HPC system eagle, make sure to start from a clean setup and type

        module purge
        module load conda        

1.  Setup and activate the Anaconda environment from a command prompt:

        conda config --add channels conda-forge
        conda install git
        git clone https://github.com/WISDEM/WEIS.git
        cd WEIS
        git checkout branch_name                         # (Only if you want to switch branches, say "develop")
        conda env create --name weis-env -f environment.yml
        conda activate weis-env                          # (if this does not work, try source activate weis-env)


2. Add in final packages and install the software

        conda install -y petsc4py mpi4py pyoptsparse     # (Mac / Linux only)
        pip install -e .

3. Instructions specific for DOE HPC system Eagle.  Before executing the setup script, do:

        module load comp-intel intel-mpi mkl
        module unload gcc
        pip install -e .

**NOTE:** To use WEIS again after installation is complete, you will always need to activate the conda environment first with `conda activate weis-env` (or `source activate weis-env`). On Eagle, make sure to reload the necessary modules

For Windows users, we recommend installing `git` and the `m264` packages in separate environments as some of the libraries appear to conflict such that WISDEM cannot be successfully built from source.  The `git` package is best installed in the `base` environment.

## Developer guide

If you plan to contribute code to WEIS, please first consult the [developer guide](https://weis.readthedocs.io/en/latest/how_to_contribute_code.html).  You are also likely to want to install some of the compiled WEIS dependencies (OpenFAST or WISDEM or pyHAMS or ROSCO) from a local copy instead of relying on the conda packages.

## Feedback

For software issues please use <https://github.com/WISDEM/WEIS/issues>.  
