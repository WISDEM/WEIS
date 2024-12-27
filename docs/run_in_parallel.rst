Run in parallel
--------------

WEIS can be run sequentially on a single processor. WEIS can also be parallelized to handle larger problems in a timely manner.


Background
------------------------------------

The parallelization of WEIS leverages the [MPI library](https://mpi4py.readthedocs.io/en/stable/). Before proceeding, please make sure that your WEIS environment includes the library mpi4py as discussed in the README.md at the root level of this repo, or on the `WEIS GitHub page <https://github.com/WISDEM/WEIS/>`_.

Parallelization in WEIS happens at two levels: 
* The first level is triggered when an optimization is run and a gradient-based optimizer is called. In this case, the OpenMDAO library will execute the finite differences in parallel. OpenMDAO then assembles all partial derivatives in a single Jacobian and iterations progress.
* The second level is triggered when multiple OpenFAST runs are specified. These are executed in parallel.
The two levels of parallelization are integrated and can co-exist as long as sufficient computational resources are available.

WEIS helps you set up the right call to MPI given the available computational resources, see the sections below.


How to run WEIS in parallel
------------------------------------

Running WEIS in parallel is slightly more elaborated than running it sequentially. A first example of a parallel call for a design optimization in WEIS is provided in example  `03_NREL5MW_OC3 <https://github.com/WISDEM/WEIS/tree/master/examples/03_NREL5MW_OC3_spar>`_. A second example runs an OpenFAST model in parallel, see example `02_run_openfast_cases <https://github.com/WISDEM/WEIS/tree/develop/examples/02_run_openfast_cases>`_. 


Design optimization in parallel
------------------------------------

These instructions follow example 03 `03_NREL5MW_OC3 <https://github.com/WISDEM/WEIS/tree/master/examples/03_NREL5MW_OC3_spar>`_. To run the design optimization in parallel, the first step consists of a pre-processing call to WEIS. This step returns the number of finite differences that are specified given the analysis_options.yaml file. To execute this step, in a terminal window navigate to example 03 and type:

.. code-block:: bash

  python weis_driver.py --preMPI=True

In the terminal the code will return the best setup for an optimal MPI call given your problem. 

If you are resource constrained, you can pass the keyword argument `--maxnP` and set the maximum number of processors that you have available. If you have 20, type:

.. code-block:: bash

  python weis_driver.py --preMPI=True --maxnP=20

These two commands will help you set up the appropriate computational resources.

At this point, you are ready to launch WEIS in parallel. If you have 20 processors, your call to WEIS will look like:

.. code-block:: bash

  mpiexec -np 20 python weis_driver.py


Parallelize calls to OpenFAST
------------------------------------

WEIS can be used to run OpenFAST simulations, such as design load cases.
[More information about setting DLCs can be found here](https://github.com/WISDEM/WEIS/blob/docs/docs/dlc_generator.rst)

To do so, WEIS is run with a single function evaluation that fires off a number of OpenFAST simulations.

Let's look at an example in `02_run_openfast_cases <https://github.com/WISDEM/WEIS/tree/develop/examples/02_run_openfast_cases>`_.

In a terminal, navigate to example 02 and type:

.. code-block:: bash
  python weis_driver_loads.py --preMPI=True

The terminal should return this message

.. code-block:: bash
  Your problem has 0 design variable(s) and 7 OpenFAST run(s)

  You are not running a design optimization, a design of experiment, or your optimizer is not gradient based. The number of parallel function evaluations is set to 1

  To run the code in parallel with MPI, execute one of the following commands

  If you have access to (at least) 8 processors, please call WEIS as:
  mpiexec -np 8 python weis_driver.py


  If you do not have access to 8 processors
  please provide your maximum available number of processors by typing:
  python weis_driver.py --preMPI=True --maxnP=xx
  And substitute xx with your number of processors

If you have access to 8 processors, you are now ready to execute your script by typing 

.. code-block:: bash
  mpiexec -np 8 python weis_driver_loads.py

If you have access to fewer processors, say 4, adjust the -np entry accordingly

.. code-block:: bash
  mpiexec -np 4 python weis_driver_loads.py