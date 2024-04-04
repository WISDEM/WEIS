This example models a land-based wind turbine rotor whose blades are equipped with trailing edge flaps. This example links back to this journal article  https://doi.org/10.1002/we.2840

Note that this example is untested and requires users to install xfoil. The path to xfoil must be provided among the modeling options. Please use with caution.  The WEIS team may not be able to support this example.

If you decide to go ahead, take a look at the file BAR_USC_flaps.yaml and especially at the field `components->blade->aerodynamic_control`. Here the geometry of the trailing edge flap is set. The yaml file models one flap (OpenFAST only supports one flap per blade) that extends between 70% and 80% of the blade span. The flap has a chordwise extension of 20% and maximum deflections of plus minus 10 degrees. WEIS models the flaps by running xfoil at the spanwise locations where the flap is located (only at those locations, outside the polars from the yaml are used) generating lookup tables at a number of flap deflections (this number can be varied by the user). The lookup tables are then passed to OpenFAST. ROSCO then has flaps capabilities that are turned on by switching flags as indicated in modeling_options.yaml.

Note that this example runs a single function call of WEIS Level 3 (OpenFAST), and does not run any optimization. The analysis_options.yaml file is empty except for specifying the path to the output folder and the base name for the output files. 
Advanced users of OpenFAST can take advantage of existing optimization capabilities for the flaps. To do so, start by looking at the analysis schema stored in weis->inputs.

