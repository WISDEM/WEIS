import os
import time
import sys
import subprocess

from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
from wisdem.inputs.validation import load_yaml

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input         = os.path.join(run_dir,"IEA-15-240-RWT_VolturnUS-S_rectangular.yaml")
fname_modeling_options = run_dir + "modeling_options_umaine_semi.yaml"
fname_analysis_options = run_dir + "analysis_options.yaml"


tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print('Run time: %f'%(time.time()-tt))
    sys.stdout.flush()

# Test that the input we are providing RAFT has not changed
this_raft_input = load_yaml(os.path.join(run_dir,'outputs','15_RAFT_Rect','raft_designs','raft_design_0.yaml'))
standard_raft_input = load_yaml(os.path.join(run_dir,'raft_input_weis.yaml'))
if this_raft_input != standard_raft_input:
    print('this_raft_input:')
    subprocess.call(['cat',os.path.join(run_dir,'outputs','15_RAFT_Rect','raft_designs','raft_design_0.yaml')])

    print('standard_raft_input:')
    subprocess.call(['cat',os.path.join(run_dir,'raft_input_weis.yaml')])
    assert(False)

# If the values have changed for a purpose, move this_raft_input to standard_raft_input and commit
