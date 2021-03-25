"""
Simple script to show how to grab all cases from a DOE run. User can then
postprocess or plot further.
"""

import os, glob
import matplotlib.pyplot as plt
import openmdao.api as om
from wisdem.commonse.mpi_tools import MPI
import numpy as np
from weis.aeroelasticse.Util.FileTools import save_yaml


# Simply gather all of the sql files
folder_output = '/scratch/dzalkind/WEIS-4/DOE/outputs_DOE_pitch_small'
optimization_logs = glob.glob(os.path.join(folder_output,'log_opt.sql_*'))

rec_data = {}
design_vars = {}
responses   = {}
iterations = []

for i, log in enumerate(optimization_logs):
    cr = om.CaseReader(log)
    cases = cr.list_cases()
    
    for casei in cases:
        iterations.append(i)
        it_data = cr.get_case(casei)

        # Collect DVs and responses separately for DOE
        for design_var in [it_data.get_design_vars()]:
            for dv in design_var:
                if i == 0:
                    design_vars[dv] = []        
                design_vars[dv].append(design_var[dv])

        for response in [it_data.get_responses()]:
            for resp in response:
                if i == 0:
                    responses[resp] = []
                responses[resp].append(response[resp])

        # parameters = it_data.get_responses()
        # Collect all parameters for convergence plots
        for parameters in [it_data.outputs]:
            for j, param in enumerate(parameters.keys()):
                if i == 0:
                    rec_data[param] = []
                rec_data[param].append(parameters[param])

# concatenate data
data_out = {}
rec_out = {}
dv_out = {}
resp_out = {}
for r in rec_data:
    rec_out[r] = np.concatenate((rec_data[r]))

for r in design_vars:
    dv_out[r] = np.concatenate((design_vars[r]))

for r in responses:
    resp_out[r] = np.concatenate((responses[r]))

data_out['design_vars'] = dv_out
data_out['responses'] = resp_out
data_out['rec_out'] = rec_out

# print(data_out)
save_yaml(folder_output,'doe_results.yaml',data_out)