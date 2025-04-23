import os
from weis import weis_main

# TEST_RUN will reduce the number and duration of simulations
TEST_RUN = False

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-3p4-130-RWT.yaml")
fname_modeling_options = os.path.join(run_dir, "iea34_modeling.yaml")
fname_analysis_options = os.path.join(run_dir, "iea34_analysis.yaml")

wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, 
                                                 fname_modeling_options, 
                                                 fname_analysis_options,
                                                 test_run=TEST_RUN
                                                 )
