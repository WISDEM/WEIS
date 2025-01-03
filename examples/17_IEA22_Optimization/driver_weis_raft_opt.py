#!/usr/bin/env python3
import os
from weis.glue_code.runWEIS import run_weis
from openmdao.utils.mpi import MPI


## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-22-280-RWT-Semi.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_raft.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_raft_ptfm_opt.yaml")

wt_opt, modeling_options, analysis_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    # shutil.copyfile(os.path.join(analysis_options["general"]["folder_output"],analysis_options["general"]["fname_output"]+".yaml"), fname_wt_input)
    print("Tower mass (kg) =", wt_opt["towerse.tower_mass"])
    print("Floating platform mass (kg) =", wt_opt["floatingse.platform_mass"])
