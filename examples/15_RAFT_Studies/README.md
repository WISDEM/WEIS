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
 Two options are provided in this example: `IEA-15-floating.yaml` (original) and `opt_22.yaml` 

 
 
