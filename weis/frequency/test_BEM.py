import os
from weis import weis_main

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "..", "examples", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S_rectangular.yaml")
fname_modeling_options = os.path.join(run_dir, "..", "..", "examples", "06_IEA-15-240-RWT", "modeling_options_umaine_semi.yaml")
fname_analysis_options = os.path.join(run_dir, "..", "..", "examples", "06_IEA-15-240-RWT", "analysis_options.yaml")

# override the modeling options so that not creating a new yaml file
modeling_override = {}
modeling_override["OpenFAST"] = {}
modeling_override["OpenFAST"]["flag"] = False
modeling_override["RAFT"] = {}
modeling_override["RAFT"]["flag"] = True
modeling_override["RAFT"]["potential_model_override"] = 0
modeling_override["RAFT"]["potential_bem_members"] = ["main_column", "column1", "column2", "column3", "Y_pontoon_lower1", "Y_pontoon_lower2", "Y_pontoon_lower3"]
modeling_override["RAFT"]["intersection_mesh"] = 1
modeling_override["RAFT"]["characteristic_length_max"] = 1
modeling_override["RAFT"]["characteristic_length_max"] = 3
modeling_override["RAFT"]["plot_designs"] = True
modeling_override["RAFT"]["save_designs"] = True

analysis_override = {}
analysis_override["general"] = {}
analysis_override["general"]["folder_output"] = run_dir
                          

wt_opt, modeling_options, opt_options = weis_main(fname_wt_input,
                                                  fname_modeling_options,
                                                  fname_analysis_options,
                                                  modeling_override=modeling_override,
                                                  analysis_override=analysis_override)
