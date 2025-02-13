import os
from weis import weis_main

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
#fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_wt_input = os.path.join(run_dir, "..", "15_RAFT_Studies", "opt_22.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_ol2cl.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_ol2cl.yaml")

wt_opt, modeling_options, analysis_options = weis_main(fname_wt_input,
                                                       fname_modeling_options,
                                                       fname_analysis_options)
