import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from openmdao.utils.mpi  import MPI
from weis.glue_code.mpi_tools import compute_optimal_nC

## File management
run_dir                 = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input          = run_dir + os.sep + 'IEA-15-240-RWT.yaml'
fname_modeling_options  = run_dir + os.sep + 'modeling_options_loads.yaml'
fname_analysis_options  = run_dir + os.sep + 'analysis_options_loads.yaml'


import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Run WEIS driver with flag prepping for MPI run.")
# Add the flag
parser.add_argument("--preMPIflag", type=bool, default=False, help="Flag for preprocessing MPI settings (True or False).")
parser.add_argument("--maxCores", type=int, default=0, help="Maximum number of cores available.")
parser.add_argument("--n_FD", type=int, default=0, help="Number of parallel finite differences, only used when MPI is turned ON.")
parser.add_argument("--n_OF_parallel", type=int, default=0, help="Number of OpenFAST calls run in parallel, only used when MPI is turned ON.")
# Parse the arguments
args = parser.parse_args()
# Use the flag in your script
if args.preMPIflag:
    print("Preprocessor flag is set to True. Running preprocessing setting up MPI run.")
else:
    print("Preprocessor flag is set to False. Run WEIS now.")

tt = time.time()
# Use the flag in your script
if args.preMPIflag:
    _, modeling_options, opt_options, n_FD, n_OF_runs = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options, prepMPI=True)
    compute_optimal_nC(n_FD, n_OF_runs, modeling_options, opt_options, max_cores = args.maxCores)
else:
    wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options, n_FD = args.n_FD, n_OF_parallel = args.n_OF_parallel)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0 and args.preMPIflag == False:
    print("Run time: %f"%(time.time()-tt))
    sys.stdout.flush()

