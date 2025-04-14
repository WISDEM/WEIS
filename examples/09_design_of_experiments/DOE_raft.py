"""
This example shows how to use the Design of Experiments (DOE) driver within
WEIS including OpenFAST.

Specifically, the DOE driver helps sample the design space of a problem, allowing
you to perform parameter sweeps or random samples of design variables. This is
useful for better understanding a design space, especially prior to running optimization.
The DOE driver is not an optimizer, but simply runs the cases prescribed.
Check out the `driver` section within the `analysis_options.yaml` for the
DOE driver settings.
"""

import os
from weis import weis_main

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input         = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = os.path.join(run_dir, "..", "15_RAFT_Studies", "modeling_options_umaine_semi.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_raft.yaml")

wt_opt, modeling_options, opt_options = weis_main(fname_wt_input,
                                                  fname_modeling_options,
                                                  fname_analysis_options)
