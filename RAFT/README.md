# RAFT - Response Amplitudes of Floating Turbines

RAFT is a Python code for frequency-domain analysis of floating wind turbines. It constitutes the "Level 1" of modeling fidelity in the [WEIS](https://weis.readthedocs.io/en/latest/index.html) toolset for floating wind turbine controls co-design. Once completed, RAFT will provide frequency-domain modeling of full floating wind turbine systems including turbine aerodynamics, controls, select structural degrees of freedom, platform hydrodynamics, and mooring response. RAFT is under active development and currently has the floating support structure components implemented. These components will be actively tested and revised as the model is verified and further developed. The mooring system is represented by NREL's new quasi-static mooring system model, [MoorPy](https://github.com/NREL/MoorPy). Upcoming development efforts focus on inclusion of tower flexibility and turbine aerodynamics, and then integration with the larger WEIS framework.

This project makes use of the HAMS (Hydrodynamic Analysis of Marine Structures) tool for boundary-element-method solution of the potential flow problem developed by Yingyi Liu and available at https://github.com/YingyiLiu/HAMS. We use a Python wrapper for it, [pyHAMS](https://github.com/WISDEM/pyHAMS).

## Getting Started

As RAFT is still a work in progress, prospective users and beta testers are recommended to begin by running runRAFT.py, which is set to use one of three included example input files.

### Prerequisites

- Python 3
- MoorPy (available at https://github.com/NREL/MoorPy)
- pyHams (available at https://github.com/WISDEM/pyHAMS)

### Installation

Download or clone this RAFT repository as well as the [MoorPy](https://github.com/NREL/MoorPy)  and [pyHAMS](https://github.com/WISDEM/pyHAMS) repositories. To install each package in development mode, go to its main directory and run ```python setup.py develop``` or ```pip install -e .``` from the command line.

## Documentation

A dedicated documentation site is under development. In the meantime, comments in the code are the best guide for how RAFT works.

## License
Licensed under the Apache License, Version 2.0
