import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from openmdao.utils.mpi  import MPI
from wisdem.inputs.validation import load_yaml

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S_rectangular.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_umaine_semi.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options.yaml")

tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print("Run time: %f"%(time.time()-tt))
    sys.stdout.flush()

# Test that the input we are providing RAFT has not changed
this_raft_input = load_yaml(os.path.join(run_dir,"outputs","15_RAFT_Rect","raft_designs","raft_design_0.yaml"))
standard_raft_input = load_yaml(os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S_raft.yaml"))
# Disable this test because we get slightly different inputs on the linux CI
# assert(this_raft_input != standard_raft_input)

# If the values have changed for a purpose, move this_raft_input to standard_raft_input and commit
