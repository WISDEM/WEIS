#!/usr/bin/env python3
import os
from weis import weis_main
import sys


## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-22-280-RWT_Floater.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_openfast.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_of_ptfm_opt.yaml")

wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, 
                                                  fname_modeling_options, 
                                                  fname_analysis_options)


