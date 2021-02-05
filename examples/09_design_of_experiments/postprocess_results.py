"""
Simple script to show how to grab all cases from a DOE run. User can then
postprocess or plot further.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd 
import multiprocessing as mp 

import openmdao.api as om

def load_OMsql(log):
    print('loading {}'.format(log))
    cr = om.CaseReader(log)
    cases = cr.list_cases()
    rec_data = {}
    iterations = []
    for i, casei in enumerate(cases):
        iterations.append(i)
        it_data = cr.get_case(casei)

        # parameters = it_data.get_responses()
        for parameters in [it_data.get_responses(), it_data.get_design_vars()]:
            for j, param in enumerate(parameters.keys()):
                if i == 0:
                    rec_data[param] = []
                rec_data[param].append(parameters[param])
                
    return rec_data


if __name__ == '__main__':
    
    # Multiprocssing?
    post_multi = True

    # sql outfile directory
    run_dir = os.path.dirname(os.path.realpath(__file__))   
    output_dir = os.path.join(run_dir, "outputs")
    doe_logs = glob.glob(os.path.join(output_dir,'log_opt.sql*'))

    # run multiprocessing
    if post_multi:
        cores = mp.cpu_count()
        pool = mp.Pool(min(len(doe_logs), cores))

        # load sql file
        outdata = pool.map(load_OMsql, doe_logs)
        pool.close()
        pool.join()
    # no multiprocessing
    else:
        outdata = [load_OMsql(doe_logs)]

    # Collect and clean up output data
    collected_data = {}
    for data in outdata:
        for key in data.keys():
            if key not in collected_data.keys():
                collected_data[key] = []

            for key_idx, _ in enumerate(data[key]):
                collected_data[key].append(np.array(data[key][key_idx]))

    df = pd.DataFrame.from_dict(collected_data)
    
    # write to file
    outdata_fname = 'doe_outdata.csv'
    outdata_fpath = os.path.join(os.getcwd(),outdata_fname) 
    df.to_csv(outdata_fpath, index=False)

