pCrunch
=======

IO and Post Processing Interface for OpenFAST Results.

:Version: 0.2.0
:Authors: Jake Nunemaker, Nikhar Abbas

Installation
------------

pCrunch is installable through pip via ``pip install pCrunch``.

Development Setup
-----------------

To set up pCrunch for development, follow these steps:

1. Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
   for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
   for the appropriate OS version.
2. From the terminal, install pip by running: ``conda install -c anaconda pip``
3. Next, create a new environment for the project with the following.

    .. code-block:: console

        conda create -n pcrunch-dev python=3.7

   To activate/deactivate the environment, use the following commands.

    .. code-block:: console

        conda activate pcrunch-dev
        conda deactivate pcrunch-dev

4. Clone the repository:
   ``git clone https://github.com/NREL/pCrunch.git``

5. Navigate to the top level of the repository
   (``<path-to-pCrunch>/pCrunch/``) and install pCrunch as an editable package
   with following commands.

    .. code-block:: console

       pip install -e '.[dev]'

6. (Preferred) Install the pre-commit hooks to autoformat code, ensuring
   consistent code style across the repository.

    .. code-block:: console

        pre-commit install

Examples
--------

For an up to date example of the core functionalities, see `example.ipynb`. More
examples coming soon.

Other Scripts
-------------

Warning: These scripts have not been updated since the move to version 0.2.0.

There are also several utility scripts provided in this repo. The HPC_tools
folder contains utility functions for working on the NREL HPC. postProcessing
contains a script for analyzing the results of multiple runs.
