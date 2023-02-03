import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

# get path to pickle and sql files
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
opt_path = mydir + os.sep + "outputs" + os.sep +"cost_bos" + os.sep + 'analysis'


# load case reader
cr = om.CaseReader(opt_path+ os.sep +"log_opt.sql")
driver_cases = cr.get_cases('driver')

# initialize
DVs = []
BOS = []

# extract
for idx, case in enumerate(driver_cases):
    dvs = case.get_design_vars(scaled=False)
    BOS.append(case.get_objectives(scaled = False)['financese.bos'][0])
    for key in dvs.keys():
        DVs.append(dvs[key])

# set length 
n_dv = len(dvs.keys())

# reshape into array
DV = np.reshape(DVs,(idx+1,n_dv),order = 'C')


#
wlf = 0.15
fcr = 0.056
c_opex = 137
c_turbine = 1115.5

results = {'DV':DV,
           'wlf':wlf,
           'fcr':fcr,
            'OC_kW':c_opex,
            'TC_kW':c_turbine,
            'BC_kW': BOS,
            'MR': 15000}

matname = 'turbinecost_newDV.mat'
savemat(matname,results)
