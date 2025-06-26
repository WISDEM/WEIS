import os
from weis.ftw.weis_wrapper import (ftw_doe, ftw_extract_doe)

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
wt_opt_doe, modeling_options_doe, analysis_options_doe = ftw_doe(
    fname_wt_input, fname_modeling_options, fname_analysis_options,
    geometry_override, modeling_override, analysis_override, TEST_RUN)

# Extract data from recorded DOE database files
input_vars, input_dataset, input_lens, input_vecs, output_vars, \
output_dataset, output_lens, output_vecs = ftw_extract_doe(
    wt_opt_doe, modeling_options_doe, analysis_options_doe)


