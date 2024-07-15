This example shows how to use the Design of Experiments (DOE) driver within WEIS, including OpenFAST.

The DOE driver helps sample the design space of a problem, allowing you to perform parameter sweeps or random samples of design variables. This is useful for better understanding a design space, especially prior to running optimization.
The DOE driver is not an optimizer, but simply runs the cases prescribed. 
Check out the `driver` section within the `analysis_options.yaml` for the DOE driver settings.

The outputs will be located in (possibly several) `log_opt.sql` files.
The `postprocess_results.py` script can be used to parse the information in those files and review the outputs.

