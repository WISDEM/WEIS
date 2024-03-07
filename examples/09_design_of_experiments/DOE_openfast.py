"""
This example shows how to use the Design of Experiments (DOE) driver within
WEIS including OpenFAST.

Specifically, the DOE driver helps sample the design space of a problem, allowing
you to perform parameter sweeps or random samples of design variables. This is
useful for better understanding a design space, especially prior to running optimization.
The DOE driver is not an optimizer, but simply runs the cases prescribed.
Check out the `driver` section within the `analysis_options.yaml` for the
DOE driver settings.
"""

from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
import os, time, sys

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
examples        = os.path.join(os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ), "examples")
fname_wt_input         = os.path.join(run_dir, "..", "06_IEA-15-240-RWT", "IEA-15-240-RWT_Monopile.yaml")
fname_modeling_options = run_dir + "modeling_options_OF.yaml"
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
