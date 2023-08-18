Getting Started
===============


Prerequisites
^^^^^^^^^^^^^

- Python 3
- The following Python packages: NumPy, MatPlotLib, yaml, scipy


Installation
^^^^^^^^^^^^

Clone the `MoorPy GitHub repository <https://github.com/NREL/MoorPy>`_.

To install for development use:

run ```python setup.py develop``` or ```pip install -e .``` from the command line in the main MoorPy directory.

To install for non-development use:

run ```python setup.py``` or ```pip install .``` from the command line in the main MoorPy directory.


Examples
^^^^^^^^

The MoorPy repository has an examples folder containing two example scripts:

- imported_system.py creates a mooring system by reading an included MoorDyn-style input file.

- manual_system.py constructs a mooring system from scratch using functions for creating each line and attachment point.

Both of these example scripts should run successfully after MoorPy has been installed.
They will produce a plot showing the mooring system in the calculated equilibrium state.
:ref:`The Model Structure page<Model Structure>` gives an overview of how a mooring system
is represented in MoorPy. :ref:`The MoorPy Usage page<MoorPy Usage>` gives more information
about the function calls used to make these examples work, as well as other available
methods and outputs of MoorPy. 