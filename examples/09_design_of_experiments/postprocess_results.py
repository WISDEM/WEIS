import openmdao.api as om

import numpy as np
import os


log_files = []
for file in os.listdir("outputs"):
    if '.sql' in file:
        log_files.append(os.path.join("outputs", file))
        
for log_file in log_files:
    cr = om.CaseReader(log_file)

    cases = cr.get_cases()

    for case in cases:
        print(case)