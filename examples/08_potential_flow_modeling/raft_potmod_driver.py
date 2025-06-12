import os
from weis import weis_main

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "nrel5mw-spar_oc3.yaml")
fname_modeling_options = os.path.join(run_dir, "raft_potmod_modeling.yaml")
fname_analysis_options = os.path.join(run_dir, "raft_potmod_analysis.yaml")

wt_opt, modeling_options, opt_options = weis_main(fname_wt_input,
                                                  fname_modeling_options,
                                                  fname_analysis_options)
