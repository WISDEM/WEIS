"""
Simple script to show how to grab all cases from a DOE run. User can then
postprocess or plot further.
"""

import glob
import os
import sys
import time

import numpy as np

import openmdao.api as om

# Simply gather all of the sql files
run_dir = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(run_dir, "outputs")
optimization_logs = glob.glob(os.path.join(output_path, "*.sql*"))

print(optimization_logs)

all_data = {}
for log in optimization_logs:
    cr = om.CaseReader(log)
    cases = cr.get_cases()
    
    for case in cases:
        for key in case.outputs.keys():
            if key not in all_data.keys():
                all_data[key] = []
            all_data[key].append(case.outputs[key])
            
            
for key in all_data.keys():
    all_data[key] = np.array(all_data[key])
    
n_cases = len(optimization_logs)

tower_mass = all_data['floatingse.tower_mass']
print(tower_mass)