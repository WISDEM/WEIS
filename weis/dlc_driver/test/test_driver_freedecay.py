import os
from weis import weis_main

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..","..","..","examples","00_setup", "ref_turbines", "IEA-15-240-RWT.yaml")
fname_modeling_options = os.path.join(run_dir, "weis_inputs", "modeling_options_freedecay.yaml")
fname_analysis_options = os.path.join(run_dir, "weis_inputs", "analysis_options_freedecay.yaml")

# Run WEIS for real now
wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, 
                                                  fname_modeling_options, 
                                                  fname_analysis_options)
