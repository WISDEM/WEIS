# WEIS

[![Coverage Status](https://coveralls.io/repos/github/WISDEM/WEIS/badge.svg?branch=develop)](https://coveralls.io/github/WISDEM/WEIS?branch=develop)
[![Actions Status](https://github.com/WISDEM/WEIS/workflows/CI_WEIS/badge.svg?branch=develop)](https://github.com/WISDEM/WEIS/actions)
[![Documentation Status](https://readthedocs.org/projects/weis/badge/?version=develop)](https://weis.readthedocs.io/en/develop/?badge=develop)


WEIS, Wind Energy with Integrated Servo-control, performs multifidelity co-design of wind turbines. WEIS is a framework that combines multiple NREL-developed tools to enable design optimization of floating offshore wind turbines.

Author: [NREL WISDEM & OpenFAST & Control Teams](mailto:systems.engineering@nrel.gov) 

## Version

This software is a version 0.0.1.

## Documentation

See local documentation in the `docs`-directory or access the online version at <https://weis.readthedocs.io/en/latest/>

## Packages

WEIS integrates in a unique workflow four models:
* [WISDEM](https://github.com/WISDEM/WISDEM) is a set of models for assessing overall wind plant cost of energy (COE).
* [OpenFAST](https://github.com/OpenFAST/openfast) is the community model for wind turbine simulation to be developed and used by research laboratories, academia, and industry.
* [TurbSim](https://www.nrel.gov/docs/fy09osti/46198.pdf) is a stochastic, full-field, turbulent-wind simulator. 
* [ROSCO](https://github.com/NREL/ROSCO) provides an open, modular and fully adaptable baseline wind turbine controller to the scientific community.

In addition, three external libraries are added:
* [ROSCO_Toolbox](https://github.com/NREL/ROSCO_toolbox) is a toolbox designed to ease controller implementation for the wind turbine researcher and tune the ROSCO controller.
* [pCrunch](https://github.com/NREL/pCrunch) is a collection of tools to ease the process of parsing large amounts of OpenFAST output data and conduct loads analysis.
* [pyOptSparse](https://github.com/mdolab/pyoptsparse) is a framework for formulating and efficiently solving nonlinear constrained optimization problems.

The core WEIS modules are:
 * _aeroelasticse_ is a wrapper to call [OpenFAST](https://github.com/OpenFAST/openfast)
 * _control_ contains the routines calling the [ROSCO_Toolbox](https://github.com/NREL/ROSCO_toolbox) and the routines supporting distributed aerodynamic control devices, such trailing edge flaps
 * _gluecode_ contains the scripts glueing together all models and libraries
 * _multifidelity_ contains the codes to run multifidelity design optimizations
 * _optimization_drivers_ contains various optimization drivers
 * _schema_ contains the YAML files and corresponding schemas representing the input files to WEIS

## Installation

On laptop and personal computers, installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WEIS requires [Anaconda 64-bit](https://www.anaconda.com/distribution/). 

The installation instructions below use the environment name, "weis-env," but any name is acceptable.

0.  On the DOE HPC system eagle, make sure to start from a clean setup and type
        
        module purge
        module load conda        

1.  Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

        conda config --add channels conda-forge
        conda create -y --name weis-env python=3.8       # (you can install an environment at a predefined path by switching "--name weis-env" with "--prefix path/weis-env")
        conda activate weis-env                          # (if this does not work, try source activate weis-env)
    
2.  Use conda to install the build dependencies.  Note the differences between Windows and Mac/Linux build systems. Skip to point #3 if you are on the DOE HPC system Eagle

        conda install -y cmake cython geopy git jsonschema make matplotlib-base numpy numpydoc openmdao openpyxl pandas pip pytest pyyaml ruamel_yaml scipy setuptools shapely six sympy swig xlrd
        conda install -y petsc4py mpi4py compilers       # (Mac / Linux only)   
        conda install -y m2w64-toolchain libpython       # (Windows only)
        pip install simpy marmot-agents jsonmerge
        git clone https://github.com/WISDEM/WEIS.git
        cd WEIS
        git checkout branch_name # (Only if you want to switch git branch, say develop)
        python setup.py develop

3. Instructions specific for DOE HPC system Eagle

        conda install -y cmake cython geopy git jsonschema make matplotlib-base numpy numpydoc openmdao openpyxl pandas pip pytest pyyaml ruamel_yaml scipy setuptools shapely six sympy swig xlrd
        conda install -y petsc4py mpi4py  
        pip install simpy marmot-agents jsonmerge
        git clone https://github.com/WISDEM/WEIS.git
        cd WEIS
        git checkout branch_name # (Only if you want to switch git branch, say develop)
        module load comp-intel intel-mpi mkl
        module unload gcc
        python setup.py develop

**NOTE:** To use WEIS again after installation is complete, you will always need to activate the conda environment first with `conda activate weis-env` (or `source activate weis-env`). On Eagle, make sure to reload the necessary modules

## Developer guide

If you plan to contribute code to WEIS, please first consult the [developer guide](https://weis.readthedocs.io/en/latest/how_to_contribute_code.html).

## Feedback

For software issues please use <https://github.com/WISDEM/WEIS/issues>.  
