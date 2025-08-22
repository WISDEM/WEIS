import os
import time
import sys

from weis     import weis_main

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = run_dir + "modeling_options_dfsm_fowt.yaml"
fname_analysis_options = run_dir + "analysis_options_dfsm_fowt.yaml"

wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, fname_modeling_options, fname_analysis_options)

