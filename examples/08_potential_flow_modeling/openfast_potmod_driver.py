import os
from weis import weis_main

TEST_RUN = False  # Set to True to reduce the number and duration of simulations

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input         = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "nrel5mw-spar_oc3.yaml")
fname_modeling_options = os.path.join(run_dir, "openfast_potmod_modeling.yaml")
fname_analysis_options = os.path.join(run_dir, "openfast_potmod_analysis.yaml")

wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, 
                                                  fname_modeling_options, 
                                                  fname_analysis_options,
                                                  test_run=TEST_RUN
                                                  )
