"""
Example showing how WEIS values can be changed programmatically in Python.

This uses the `overridden_values` dict given to `run_weis`.
Specifically, you can supply a dictionary of values to overwrite after
setup is called.
This feature hasn't been tested and confirmed working for all values,
especially those that OpenFAST uses.
"""

from weis.glue_code.runWEIS import run_weis
import os, time, sys

## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
fname_wt_input         = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "06_IEA-15-240-RWT", "IEA-15-240-RWT.yaml")
fname_modeling_options = run_dir + "modeling_options.yaml"
fname_analysis_options = run_dir + "analysis_options.yaml"

# Run the base simulation with no changes to the inputs
wt_opt, modeling_options, opt_options = run_weis(
    fname_wt_input, fname_modeling_options, fname_analysis_options
)
print(f"Tip deflection: {wt_opt['rotorse.rs.tip_pos.tip_deflection'][0]} meters")


# Construct a dict with values to overwrite
overridden_values = {}
overridden_values["rotorse.wt_class.V_mean"] = 11.5

# Run the modified simulation with the overwritten values
wt_opt, modeling_options, opt_options = run_weis(
    fname_wt_input,
    fname_modeling_options,
    fname_analysis_options,
    overridden_values=overridden_values,
)
print(f"Tip deflection: {wt_opt['rotorse.rs.tip_pos.tip_deflection'][0]} meters")
