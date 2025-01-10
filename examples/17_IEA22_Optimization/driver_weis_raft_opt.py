#!/usr/bin/env python3
import os
import shutil
from weis.glue_code.runWEIS import run_weis
import wisdem.inputs as sch
import numpy as np
from openmdao.utils.mpi import MPI


## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

fname_wt_input = mydir + os.sep + "IEA-22-280-RWT-Semi.yaml"
# fname_wt_input = '/Users/dzalkind/Projects/IEA-22MW/FloaterIEA22/33_Round_DesRo1/IEA-22-280-RWT.yaml'
# fname_wt_input = os.path.join(mydir,'11_Lower_Pontoon_Stiffeners','IEA-22-280-RWT.yaml')
fname_modeling_options = mydir + os.sep + "modeling_options_raft.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_raft_ptfm_opt.yaml"
wt_opt, modeling_options, analysis_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    # shutil.copyfile(os.path.join(analysis_options['general']['folder_output'],analysis_options['general']['fname_output']+'.yaml'), fname_wt_input)
    print("Tower mass (kg) =", wt_opt["towerse.tower_mass"])
    print("Floating platform mass (kg) =", wt_opt["floatingse.platform_mass"])
