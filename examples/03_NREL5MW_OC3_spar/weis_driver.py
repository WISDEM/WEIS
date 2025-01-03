import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from weis.glue_code.weis_args   import weis_args, get_max_procs, set_modopt_procs
from openmdao.utils.mpi  import MPI

# Parse args
args = weis_args()

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, 'nrel5mw-spar_oc3.yaml')
fname_modeling_options = os.path.join(run_dir, 'modeling_options.yaml')
fname_analysis_options = os.path.join(run_dir, 'analysis_options.yaml')

tt = time.time()
maxnP = get_max_procs(args)

modeling_override = None
if MPI:
    # Pre-compute number of cores needed in this run
    _, modeling_options, _ = run_weis(fname_wt_input,
                                    fname_modeling_options, 
                                    fname_analysis_options, 
                                    prepMPI=True, 
                                    maxnP = maxnP)

    modeling_override = set_modopt_procs(modeling_options)

# Run WEIS for real now
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, 
                                                    fname_modeling_options, 
                                                    fname_analysis_options,
                                                    modeling_override=modeling_override,
                                                    prepMPI=args.preMPI)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0 and args.preMPI == False:
    print("Run time: %f"%(time.time()-tt))
    sys.stdout.flush()
