pCrunch
=======

IO and Post Processing Interface for OpenFAST Results.

:Authors: Jake Nunemaker, Nikhar Abbas

Installation
------------

pCrunch is installable through pip via ``pip install pCrunch`` or conda, ``conda install pCrunch``.

Development Setup
-----------------

To set up pCrunch for development, follow these steps:

1. Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
   for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
   for the appropriate OS version.
2. From the terminal, install pip by running: ``conda install -c anaconda pip``
3. Next, create a new environment for the project with the following.

    .. code-block:: console

        conda create -n pcrunch-dev

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
