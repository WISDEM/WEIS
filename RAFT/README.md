# RAFT - Response Amplitudes of Floating Turbines

RAFT is a Python code for frequency-domain analysis of floating wind turbines. It constitutes the "Level 1" of modeling fidelity in the [WEIS](https://weis.readthedocs.io/en/latest/index.html) toolset for floating wind turbine controls co-design. RAFT provides frequency-domain modeling of full floating wind turbine systems including turbine aerodynamics, controls, rigid-body platform hydrodynamics, and mooring response. The rotor aerodynamics are provided by [CCBlade](https://github.com/WISDEM/CCBlade) and the mooring system is represented by NREL's new quasi-static mooring system model, [MoorPy](https://github.com/NREL/MoorPy). Potential-flow hydrodynamics can optionally be used via [pyHAMS](https://github.com/WISDEM/pyHAMS), a Python wrapper of the HAMS (Hydrodynamic Analysis of Marine Structures) tool for boundary-element-method solution of the potential flow problem developed by [Yingyi Liu](https://github.com/YingyiLiu/HAMS).

RAFT v1.0.0 includes the capabilities described above, and further development is planned to expand these capabilities. Documentation and verification efforts are ongoing. Please see [RAFT's documentation](https://openraft.readthedocs.io/en/latest/) for more information.


## Getting Started

New users of RAFT as a standalone model are recommended to begin by looking at the input file and script provided in the examples folder, and seeking further information from the [RAFT documentation](https://openraft.readthedocs.io/en/latest/). For use as part of the WEIS toolset, information will be provided once this capability is completed in the [WEIS documentation](https://weis.readthedocs.io/en/latest/).


### Prerequisites

- Python 3
- NumPy
- Matplotlib
- SciPy
- YAML
- MoorPy (available at https://github.com/NREL/MoorPy)
- pyHams (available at https://github.com/WISDEM/pyHAMS)
- CCBlade and WISDEM* (available at https://github.com/WISDEM/WISDEM)

\* RAFT uses CCBlade and currently requires additional related functions from the larger WISDEM code. We recommend installing WISDEM for the time being.

### Installation

Download/clone and install this RAFT repository as well as that of [MoorPy](https://github.com/NREL/MoorPy), [pyHAMS](https://github.com/WISDEM/pyHAMS), and [WISDEM](https://github.com/WISDEM/WISDEM). To install RAFT in development mode, go to its directory and run ```python setup.py develop``` or ```pip install -e .``` from the command line.

## Documentation and Issues

Please see <https://weis.readthedocs.io/en/latest/> for documentation.

Questions and issues can be posted at <https://github.com/WISDEM/RAFT/issues>.

## License
RAFT is licensed under the Apache License, Version 2.0

