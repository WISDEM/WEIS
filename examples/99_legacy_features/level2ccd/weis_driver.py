import os 
from weis import weis_main

if __name__ == "__main__":

    ## File management
    run_dir = os.path.dirname( os.path.realpath(__file__) )
    fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S.yaml")
    fname_modeling_options = os.path.join(run_dir, "modeling_options.yaml")
    fname_analysis_options = os.path.join(run_dir, "analysis_options.yaml")
    
    wt_opt, modeling_options, opt_options = weis_main(fname_wt_input,
                                                      fname_modeling_options,
                                                      fname_analysis_options)
    

