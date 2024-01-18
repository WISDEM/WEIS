
from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
import os, time, sys

'''
This example uses WEIS to generate the platform model and run OpenFAST simulations
The OpenFAST simulations have not finished being verified against the original model
'''

## File management
run_dir                 = os.path.dirname( os.path.realpath(__file__) )
wisdem_examples         = os.path.join(os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ), "WISDEM", "examples")
fname_wt_input          = run_dir + os.sep + 'rm1_ptfm_opt.yaml'
fname_modeling_options  = run_dir + os.sep + 'modeling_options_MHK_WISDEM.yaml'
fname_analysis_options  = run_dir + os.sep + 'analysis_options_control_opt.yaml'


tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print("Run time: %f"%(time.time()-tt))
    sys.stdout.flush()