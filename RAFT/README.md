# RAFT - Response Amplitudes of Floating Turbines

RAFT is a Python code for frequency-domain analysis of floating wind turbines. It constitutes the "Level 1" of modeling fidelity in the [WEIS](https://weis.readthedocs.io/en/latest/index.html) toolset for floating wind turbine controls co-design. RAFT provides frequency-domain modeling of full floating wind turbine systems including turbine aerodynamics, controls, rigid-body platform hydrodynamics, and mooring response. The rotor aerodynamics are provided by [CCBlade](https://github.com/WISDEM/CCBlade) and the mooring system is represented by NREL's new quasi-static mooring system model, [MoorPy](https://github.com/NREL/MoorPy). Potential-flow hydrodynamics can optionally be used via [pyHAMS](https://github.com/WISDEM/pyHAMS), a Python wrapper of the HAMS (Hydrodynamic Analysis of Marine Structures) tool for boundary-element-method solution of the potential flow problem developed by [Yingyi Liu](https://github.com/YingyiLiu/HAMS).

RAFT v1.0.0 includes the capabilities described above, and further development is planned to expand these capabilities. Documentation and verification efforts are ongoing. Please see [RAFT's documentation](https://openraft.readthedocs.io/en/latest/) for more information.


## Getting Started

New users of RAFT as a standalone model are recommended to begin by looking at the input file and script provided in the examples folder, and seeking further information from the [RAFT documentation](https://openraft.readthedocs.io/en/latest/). For use as part of the WEIS toolset, information will be provided once this capability is completed in the [WEIS documentation](https://weis.readthedocs.io/en/latest/). For now, the following will help get started with running RAFT as a standalone model.

RAFT uses a number of prerequisites, or other python package dependencies, to run its calculations. The most notable ones are listed below:

- Python 3
- NumPy
- Matplotlib
- SciPy
- YAML
- MoorPy (available at https://github.com/NREL/MoorPy)
- pyHams (available at https://github.com/WISDEM/pyHAMS)
- CCBlade or WISDEM* (https://github.com/WISDEM/CCBlade or https://github.com/WISDEM/WISDEM)

\* CCBlade is a module of WISDEM, but can be used separately. RAFT only requires CCBlade (and some additional related functions) out of the larger WISDEM code. New users can install either CCBlade or WISDEM, but for highest efficiency, we recommend installing CCBlade, without the entire WISDEM installation.

To install all required python packages to run RAFT, follow the steps below.

1. Install most of the required python packages to run RAFT from the "requirements" yaml.

        (base) PS ANY_PATH> conda env create -f raft-env.yaml

    This will create a new python virtual environment on a new user's local machine called "raft-env"

2. Activate the new virtual environment

        (base) PS ANY_PATH> conda activate raft-env

    This will activate the newly created virtual environment, where we will install the remaining dependencies

3. Install the RAFT package
    - Navigate to a directory of your choosing on your local machine and clone this RAFT repository to that new directory

            (raft-env) PS YOUR_PATH> git clone https://github.com/WISDEM/RAFT.git
    
    - Staying in the same directory, install the RAFT package in it's "editable" mode

            (raft-env) PS YOUR_PATH/RAFT> pip install -e .

4. Repeat Step 3 two more times, one for CCBlade, and one for MoorPy

        (raft-env) PS YOUR_PATH> git clone https://github.com/WISDEM/CCBlade.git
        (raft-env) PS YOUR_PATH/CCBlade> pip install -e .
        (raft-env) PS YOUR_PATH/CCBlade> cd ..
        (raft-env) PS YOUR_PATH> git clone https://github.com/NREL/MoorPy.git
        (raft-env) PS YOUR_PATH/MoorPy> pip install -e .

    ** If you are running into errors with installing CCBlade, make sure there is not a file called "_bem.cp39-win_amd64.pyd" inside of CCBlade/ccblade. Deleting this file will allow the above commands to run.


This new raft-env should now be compatible to run RAFT standalone. Dependencies like CCBlade and MoorPy are still under development, which is why for now, it will be easier to install them in their editable forms.

The other main dependency, PyHAMS, is included within the raft-env.yaml file and is installed in Step 1.

Another point to note is that ```python setup.py develop``` has become outdated, and ```pip install -e .``` is preferred.

If you need to remove any virtual environment for any reason, you can run 

    conda env remove -n "name-of-the-virtual-environment"



## Documentation and Issues

Please see <https://weis.readthedocs.io/en/latest/> for documentation.

Questions and issues can be posted at <https://github.com/WISDEM/RAFT/issues>.

## License
RAFT is licensed under the Apache License, Version 2.0

