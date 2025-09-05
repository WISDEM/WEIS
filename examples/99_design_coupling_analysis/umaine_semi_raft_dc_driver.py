import os
import numpy as np
from weis.ftw.weis_wrapper import ftw_doe
from weis.ftw.surrogate    import ftw_surrogate_modeling
from wisdem.inputs import load_yaml, write_yaml

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

# # Run DOE to prepare for the surrogate model training
# doedata, fname_doedata, fname_smt, skip_training_if_sm_exist = ftw_doe(
#     fname_wt_input, fname_modeling_options, fname_analysis_options,
#     geometry_override, modeling_override, analysis_override, TEST_RUN)

# How can we get the outputs from our own DOE data and problem vars?

fname_doedata = os.path.join(run_dir, "log_opt.sql-doedata.yaml")
doedata = load_yaml(fname_doedata)

# Train WTSM
WTSM = ftw_surrogate_modeling(
    fname_doedata=fname_doedata, 
    fname_smt=os.path.join(run_dir, 'test.smt'),
    doedata=doedata, 
    WTSM=None, 
    skip_training_if_sm_exist=False
    )

# Usage Example (Temporary code --- to be removed)
input_bounds = WTSM.get_input_bounds()
input_lower = input_bounds[0,:].reshape(1,-1)
input_upper = input_bounds[1,:].reshape(1,-1)
x_normalized = np.random.rand(1,input_lower.size)
x = input_lower + (input_upper - input_lower)*x_normalized
y, v = WTSM.predict(x)
print(y)
print(v)
