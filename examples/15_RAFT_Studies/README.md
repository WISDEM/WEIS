In this example, we can optimize the UMaine semi using RAFT, and run a DLC analysis on the initial/final geometries.

# RAFT Optimization
To optimize the UMaine semi using raft, run
 ```
 python weis_driver_umaine_semi.py
 ```
 
 In `analysis_options.yaml`, the design variables, constraints, merit figures, and optimization solver can be set up.  
 Increase `max_iter` to run more iterations.
 Groups of joints can be simultaneously set as design variables, e.g., when we vary the draft in the `z_coordinate` design variable.
 The diameter and thickness of members can also be grouped and set to `constant: True` if we want constant diameter/thickenss columns.
 
 # OpenFAST DLC runs
 To run a DLC analysis using WEIS, run
  ```
 python weis_driver_level3.py
  ```
  
 The type of DLCs that are run can be set in `modeling_options_level3.yaml`, along with the metocean conditions.
 Set the geometry used in WEIS by setting `fname_wt_input` in `weis_driver_level3.py`. 
 Two options are provided in this example: `../06_IEA-15-240-RWT/IEA-15-240-RWT_VolturnUS-S.yaml` (original) and `raft_opt_out.yaml`, the optimized output.

 
# RAFT Surrogate Model Training with Design of Experiment
To create the surrogate model with RAFT simulations, run
 ```
 python weis_driver_level1_doe.py
 ```
 If parallel execution is possible (e.g., HPC) use slurm script to run with MPI across large number of cpus/nodes
 ```
 sbatch weis_driver_level1_doe.sh
 ```
 or MPI directly for cpus in a local computer
 ```
 mpirun -np 4 python weis_driver_level1_doe.py
 ```

 In `analysis_options_level1_doe.yaml`, the design variables (control and plant), driver (optimization:flag: False, design_of_experiments:flag: True), and recorder (flag: True, includes: ['*']) can be set up. Variables marked as design_variables (flag: True) are used as inputs, while all other variables are used as outputs of the surrogate model. Not all parameters are supported at this moment. Seven control parameters, diameters of floating platform members, joint locations, rotor diameter are currently implemented.
 
 Once design of experiment is completed, a surrogate model for each output variable will be trained (in parallel if MPI is used), and recorder file_name.smt (log_opt.smt in the tutorial case) will be created.

 At this moment, surrogate model training is not stable with large number of samples.

 Further design coupling studies and/or surrogate-based optimization studies can be done in separate script files to be developed.
 
