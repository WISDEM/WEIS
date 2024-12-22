import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from openmdao.utils.mpi  import MPI

## File management
run_dir                 = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input          = os.path.join(run_dir, "nrel5mw-spar_oc3.yaml")
fname_modeling_options  = os.path.join(run_dir, 'modeling_options.yaml')
fname_analysis_options  = os.path.join(run_dir, 'analysis_options.yaml')

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Run WEIS driver with flag prepping for MPI run.")
# Add the flag
parser.add_argument("--preMPI", type=bool, default=False, help="Flag for preprocessing MPI settings (True or False).")
parser.add_argument("--maxnP", type=int, default=0, help="Maximum number of cores available.")
parser.add_argument("--nFD", type=int, default=0, help="Number of parallel finite differences, only used when MPI is turned ON.")
parser.add_argument("--nOFp", type=int, default=0, help="Number of OpenFAST calls run in parallel, only used when MPI is turned ON.")
# Parse the arguments
args = parser.parse_args()
# Use the flag in your script
if args.preMPI:
    print("Preprocessor flag is set to True. Running preprocessing setting up MPI run.")
else:
    print("Preprocessor flag is set to False. Run WEIS now.")

tt = time.time()
# Use the flag in your script
if args.preMPI:
    _, _, _ = run_weis(fname_wt_input, 
                       fname_modeling_options, 
                       fname_analysis_options, 
                       prepMPI=True, 
                       maxnP = args.maxnP)
else:
    _, modeling_options, _ = run_weis(fname_wt_input,
                                      fname_modeling_options, 
                                      fname_analysis_options, 
                                      prepMPI=True, 
                                      maxnP = args.maxnP)
    modeling_override = {}
    modeling_override['General'] = {}
    modeling_override['General']['openfast_configuration'] = {}
    modeling_override['General']['openfast_configuration']['nFD'] = modeling_options['General']['openfast_configuration']['nFD']
    modeling_override['General']['openfast_configuration']['nOFp'] = modeling_options['General']['openfast_configuration']['nOFp']
    wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, 
                                                     fname_modeling_options, 
                                                     fname_analysis_options,
                                                     modeling_override=modeling_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0 and args.preMPI == False:
    print("Run time: %f"%(time.time()-tt))
    sys.stdout.flush()
