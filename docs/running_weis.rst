Running WEIS
============

The WEIS Driver
----------------

The WEIS driver is the top level script that runs the WEIS software. See examples in `the examples folder <https://github.com/WISDEM/WEIS/tree/main/examples>`_ for how to set up and run WEIS. The WEIS driver takes three input files, see :ref:`inputs-documentation` for more details on the input files.The options in the input files can be overridden by providing them as options to WEIS driver. 
This allows you to change the inputs and options without modifying the input files, which is useful for debugging and parametric studies. See example below:

.. literalinclude:: ../weis/test/test_overrides_driver.py
   :language: python
   :linenos:
   :lines: 44-51


Starting optimizations from previous runs
------------------------------------------

There are several ways that you can restart your optimization from previous runs.
1. You can take the geometry yaml output from a previous run and use it as the geometry input for your current run. This is more general and can be used for any type of optimization, but it requires you to have the geometry yaml outputs from the previous run. Additionally, you do not retain the optimization history.
2. If you are using a pyoptsparse driver (e.g., SNOPT), you can output your optimization history file. In subsequent runs, add the option of *hotstart_file* in your analysis inputs and points to the optimization history file you output previously. This will allow you to start your optimization from the previous run. Note that this is only available for pyoptsparse driver.



Running WEIS in parallel
-------------------------

WEIS can be run sequentially on a single processor. WEIS can also be parallelized to handle larger problems in a timely manner.

The parallelization of WEIS leverages the [MPI library](https://mpi4py.readthedocs.io/en/stable/). To run WEIS in parallel, make sure that your WEIS environment includes the library mpi4py as discussed in the README.md at the root level of this repo, or on the `WEIS GitHub page <https://github.com/WISDEM/WEIS/>`_.

When mpi4py is available, running WEIS in parallel is as simple as executing this command:

.. code-block:: bash

  mpiexec -np 20 python weis_driver.py

Where you can adjust the number of processors available (`-np`) based on your computational resources. The lower limit for `-np` is 2, and there is no upper limit, as WEIS allocates processes according to its internal logic. In the worst case, WEIS might simply not use the full resources made available by the user.

Advanced users and developers, the next paragraphs document the parallelization logic.

Advanced Users and Developers
------------------------------------

Parallelization in WEIS happens at two levels: 
* The first level is triggered when an optimization is run and a gradient-based optimizer is called. In this case, the OpenMDAO library will execute the finite differences in parallel. OpenMDAO then assembles all partial derivatives in a single Jacobian and iterations progress.
* The second level is triggered when multiple OpenFAST runs are specified. These are executed in parallel.
The two levels of parallelization are integrated and can co-exist as long as sufficient computational resources are available.

Based on the number of processors available, WEIS allocates the processors to either handle the finite differences or the OpenFAST calls. The allocation is performed by running WEIS twice. The two calls happen in the script `main.py <https://github.com/WISDEM/WEIS/blob/develop/weis/main.py>`_. The first call to WEIS is performed by passing the keyword argument `preMPI` set to True. Here, WEIS sets up the OpenMDAO problem and returns the number of design variables and OpenFAST calls. The two integers are stored among the `modeling_options`. The second call of WEIS allocates the processors based on these two numbers. Note that advanced users can ask WEIS to return some print statements describing the problem size and the allocation of the resources given the number of processors. To do so, users need to set the `maxnP` keyword argument.

.. code-block:: bash
  _, modeling_options, _ = run_weis(fname_wt_input,
                                    fname_modeling_options, 
                                    fname_analysis_options, 
                                    prepMPI=True, 
                                    maxnP=maxnP)

The allocation of processors follows a color mapping scheme. Let's explain it with an example. Say the design optimization has 8 design variables and 10 OpenFAST simulations. The maximum level of parallelization will use 8+8x10=88 processors. The color mapping logic will make sure that 8 processors are passed to OpenMDAO to handle the finite differencing and build the Jacobian, whereas the other 80 processors will be sitting and waiting to receive the 80 OpenFAST simulations, 10 per design variable. If the user allocates more than 88 processors, all the extra processors will sit and do nothing, until WEIS completes its run. If the user allocates less than 88 processors, WEIS will first stack some OpenFAST runs sequentially. If less than 16 processors are available, WEIS will start allocating less than 8 processors for the Jacobian. Note that this setup is robust, but is suboptimal, as at any given time either the top 8 master processors are running and the bottom 80 OpenFAST processors are idling, or viceversa the top 8 master processors are idling and the bottom 80 OpenFAST processors are running. The suboptimality is highest for problems with many design variables and a single OpenFAST run. In this latter case, at any given time only 50% of the resources are used at any given time. The logic however works nicely for problems with several OpenFAST runs, which represent the biggest computational burden of WEIS and therefore benefit greatly from being parallelized.
