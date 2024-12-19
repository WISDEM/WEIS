Run in parallel
--------------

WEIS can be run sequentially on a single processor. WEIS can however also be parallelized to handle larger problems in a timely manner.


Background
------------------------------------

The parallelization of WEIS leverages the [MPI library](https://mpi4py.readthedocs.io/en/stable/). Before proceeding, please make sure that your WEIS environemnt includes the library mpi4py as discussed in the README.md at the root level of this repo, or on the `WEIS GitHub page <https://github.com/WISDEM/WEIS/>`_.

Parallelization in WEIS happens at two levels: 
* The first level is triggered when an optimization is run and a gradient-based optimizer is called. In this case, the OpenMDAO library will execute the finite differences in parallel. OpenMDAO then assembles all partial derivatives in a single Jacobian and iterations progress.
* The second level is triggered when multiple OpenFAST runs are specified. These are executed in parallel.
The two levels of parallelization are integrated and can co-exist as long as sufficient computational resources are available.

WEIS helps you set up the right call to MPI given the available computational resources, see the section below.


How to run WEIS in parallel
------------------------------------

Running WEIS in parallel is slightly more elaborated than running it sequentially. An example of a parallel call to WEIS is provided in example  `03_NREL5MW_OC3 <https://github.com/WISDEM/WEIS/tree/master/examples/03_NREL5MW_OC3_spar>`_ 

The first consists of a pre-processing call to WEIS. This step returns the number of finite differences that are specified given the analysis_options.yaml file. To execute this step, in a terminal window type:

.. code-block:: bash

  python weis_driver.py --preMPIflag=True

In the terminal the code will return the best setup for an optimal MPI call given your problem. 

If you are resource constrained, you can pass the keyword argument `--maxCores` and set the maximum number of processors that you have available. If you have 20, type:

.. code-block:: bash

  python weis_driver.py --preMPIflag=True --maxCores=20

These two commands will help you set the three keyword arguments `nC`, `n_FD`, and `n_OFp`.

At this point, you are ready to launch WEIS in parallel. Assume that `nC=12`, `n_FD=2`, and `n_OFp=5`, your call to WEIS will look like:

.. code-block:: bash

  mpirun -np 12 python runWEIS.py --n_FD=2 --n_OF_parallel=5

Lastly, note that the two steps can be coded up in a batch script that automatically reads `nC`, `n_FD`, and `n_OFp` from the preprocessing of WEIS and passes to the full call of WEIS. 
You can do this by listing these lines of code in a batch script:
.. code-block:: bash

  source activate weis-env    
  output=$(python weis_driver.py --preMPIflag=True --maxCores=104)    
  nC=$(echo "$output" | grep 'nC=' | awk -F'=' '{print $2}')
  n_FD=$(echo "$output" | grep 'n_FD=' | awk -F'=' '{print $2}')
  n_OFp=$(echo "$output" | grep 'n_OFp=' | awk -F'=' '{print $2}')
  mpirun -np $nC python runWEIS.py --n_FD=$n_FD --n_OF_parallel=$n_OFp    


You can see this example in the `example sbatch script for NREL's HPC system Kestrel <https://github.com/WISDEM/WEIS/tree/master/examples/03_NREL5MW_OC3_spar/sbatch_hpc.sh>`.