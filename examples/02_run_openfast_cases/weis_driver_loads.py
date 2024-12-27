import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from openmdao.utils.mpi  import MPI

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, 'IEA-15-240-RWT.yaml')
fname_modeling_options = os.path.join(run_dir, 'modeling_options_loads.yaml')
fname_analysis_options = os.path.join(run_dir, 'analysis_options_loads.yaml')

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Run WEIS driver with flag prepping for MPI run.")
# Add the flag
parser.add_argument("--preMPI", type=bool, default=False, help="Flag for preprocessing MPI settings (True or False).")
parser.add_argument("--maxnP", type=int, default=0, help="Maximum number of processors available.")
# Parse the arguments
args = parser.parse_args()
# Use the flag in your script
if args.preMPI:
    print("Preprocessor flag is set to True. Running preprocessing setting up MPI run.")
else:
    print("Preprocessor flag is set to False. Run WEIS now.")

tt = time.time()

# Set max number of processes, either set by user or extracted from MPI
if args.preMPI:
    maxnP = args.preMPI
else:
    if MPI:
        maxnP = MPI.COMM_WORLD.Get_size()
    else:
        maxnP = 1

if args.preMPI:
    _, _, _ = run_weis(fname_wt_input, 
                       fname_modeling_options, 
                       fname_analysis_options, 
                       prepMPI=True, 
                       maxnP = maxnP)
else:
    if MPI:
        _, modeling_options, _ = run_weis(fname_wt_input,
                                        fname_modeling_options, 
                                        fname_analysis_options, 
                                        prepMPI=True, 
                                        maxnP = maxnP)

        modeling_override = {}
        modeling_override['General'] = {}
        modeling_override['General']['openfast_configuration'] = {}
        modeling_override['General']['openfast_configuration']['nFD'] = modeling_options['General']['openfast_configuration']['nFD']
        modeling_override['General']['openfast_configuration']['nOFp'] = modeling_options['General']['openfast_configuration']['nOFp']
    else:
        modeling_override = None
    wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, 
                                                     fname_modeling_options, 
                                                     fname_analysis_options,
                                                     modeling_override=modeling_override)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0 and args.preMPI == False:
    print("Run time: %f"%(time.time()-tt))
    sys.stdout.flush()
