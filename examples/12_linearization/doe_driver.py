import os, time, sys
from weis.glue_code.runWEIS     import run_weis


## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
wisdem_examples        = os.path.join(os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ), "WISDEM", "examples")
fname_wt_input         = os.path.join(wisdem_examples, "09_floating", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = run_dir + os.sep + "modeling_options_doe.yaml"
fname_analysis_options = run_dir + os.sep + "analysis_options_doe.yaml"

wt_opt, modeling_options, analysis_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)
