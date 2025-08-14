import os
from weis.ftw.weis_wrapper import ftw_doe
from weis.ftw.

# TEST_RUN will reduce the number and duration of simulations
TEST_RUN = False

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S_rectangular.yaml")
fname_modeling_options = os.path.join(run_dir, "umaine_semi_raft_dc_modeling.yaml")
fname_analysis_options = os.path.join(run_dir, "umaine_semi_raft_dc_analysis.yaml")
geometry_override = {}
modeling_override = {}
analysis_override = {}

# Run DOE to prepare for the surrogate model training
doedata, fname_doedata, fname_sm, skip_training_if_sm_exist = ftw_doe(
    fname_wt_input, fname_modeling_options, fname_analysis_options,
    geometry_override, modeling_override, analysis_override, TEST_RUN)

# Train SM
smdata = ftw_sm_train(doedata=doedata, sm=None,
    fname_doedata=fname_doedata, fname_sm=fname_sm,
    skip_training_if_sm_exist = skip_training_if_sm_exist)