
from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
import os, time, sys

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
wisdem_examples        = os.path.join(os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ), "WISDEM", "examples")
fname_wt_input         = os.path.join(wisdem_examples,'09_floating/IEA-15-240-RWT_VolturnUS-S.yaml')
# fname_wt_input         = '/Users/dzalkind/Tools/WEIS-4/examples/15_RAFT_Studies/IEA-15-floating.yaml'
fname_modeling_options = run_dir + "modeling_options_level3.yaml"
fname_analysis_options = run_dir + "analysis_options_level3.yaml"


tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print('Run time: %f'%(time.time()-tt))
    sys.stdout.flush()
