"""
Example showing how WEIS values can be changed programmatically in Python.

This uses the `overridden_values` dict given to `run_weis`.
Specifically, you can supply a dictionary of values to overwrite after
setup is called.
This feature hasn't been tested and confirmed working for all values,
especially those that OpenFAST uses.
"""

import os
from weis import weis_main

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "..", "examples", "00_setup", "ref_turbines", "IEA-3p4-130-RWT.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options.yaml")

# Run the base simulation with no changes to the inputs
wt_opt, modeling_options, opt_options = weis_main(
    fname_wt_input, fname_modeling_options, fname_analysis_options
)
print(f"Tip deflection: {wt_opt['rotorse.rs.tip_pos.tip_deflection'][0]} meters")


# Construct a dict with values to override.
# The dict should match the wt_opt openmdao problem
geometry_override = {}
geometry_override["rotorse.wt_class.V_mean"] = 11.5

# For modeling and analysis, the structure should match the input yaml structure
modeling_override = {}
modeling_override["WISDEM"] = {}
modeling_override["WISDEM"]["TowerSE"] = {}
modeling_override["WISDEM"]["TowerSE"]["gamma_b"] = 1.25

analysis_override = {}
analysis_override["general"] = {}
analysis_override["general"]["folder_output"] = "override_output"


# Run the modified simulation with the overwritten values
wt_opt, modeling_options, opt_options = weis_main(
    fname_wt_input,
    fname_modeling_options,
    fname_analysis_options,
    geometry_override=geometry_override,
    modeling_override=modeling_override,
    analysis_override=analysis_override,
)
print(f"Tip deflection: {wt_opt['rotorse.rs.tip_pos.tip_deflection'][0]} meters")
