import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI

run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input         = os.path.join(run_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = run_dir + "modeling_options_level1_doe.yaml"
fname_analysis_options = run_dir + "analysis_options_level1_doe.yaml"
overridden_values = {}

tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options, overridden_values)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print('Run time: %f'%(time.time()-tt))
    sys.stdout.flush()

print('rank = {:}, exiting'.format(rank))