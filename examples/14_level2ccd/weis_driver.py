import os 
from weis.glue_code.runWEIS     import run_weis


if __name__ == "__main__":

    ## File management
    run_dir = os.path.dirname( os.path.realpath(__file__) )
    fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-floating.yaml")
    fname_modeling_options = os.path.join(run_dir, "modeling_options.yaml")
    fname_analysis_options = os.path.join(run_dir, "analysis_options.yaml")
    
    wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)
    

