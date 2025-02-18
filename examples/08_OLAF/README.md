This example runs the free-wake vortex solver OLAF in OpenFAST. This is done by switching the AeroDyn15 flag `Wake_Mod` to 3. Next, multiple OLAF-specific inputs are specified.

Note that this example runs a single function call of WEIS Level 3 (OpenFAST), and does not run any optimization. The analysis_options.yaml file is empty except for specifying the path to the output folder and the base name for the output files. 
Advanced users of OpenFAST can take advantage of existing optimization capabilities. To do so, start by looking at the analysis schema stored in weis->inputs.
