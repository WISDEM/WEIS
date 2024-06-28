import os 
from weis.glue_code.runWEIS     import run_weis


if __name__ == "__main__":
    
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    
    # get path to modelling, geometry and analysis files
    fname_modeling_options = mydir + os.sep + "modeling_options_RM1.yaml"
    fname_wt_input   = mydir + os.sep + "RM1.yaml"
    fname_analysis_options      = mydir + os.sep + "analysis_options_loads.yaml"
    
    wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)
    