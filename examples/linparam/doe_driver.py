import os
from weis.glue_code.runWEIS     import run_weis

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input         = os.path.join(os.path.dirname(mydir),"06_IEA-15-240-RWT","IEA-15-240-RWT.yaml")
fname_modeling_options = mydir + os.sep + "modeling_options_doe.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_doe.yaml"

wt_opt, modeling_options, analysis_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)
