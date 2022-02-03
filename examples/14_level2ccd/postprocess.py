import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt

# get path to pickle and sql files
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
opt_path = mydir + os.sep + "outputs" + os.sep +"BV_constraints_single"
pkl_path = opt_path + os.sep+  "ABCD_matrices.pkl"

# load pickle file
with open(pkl_path, 'rb') as handle:
    ABCD_list = pickle.load(handle)
  
# load case reader
cr = om.CaseReader(opt_path+ os.sep +"log_opt.sql")
driver_cases = cr.get_cases('driver')

# initialize
DVs = []
LCOE = []

# load design variables
for idx, case in enumerate(driver_cases):
    dvs = case.get_design_vars(scaled=False)
    LCOE.append(case.get_objectives(scaled = False)['financese_post.lcoe'][0])
    for key in dvs.keys():
        DVs.append(dvs[key])
    
# set length 
n_dv = len(dvs.keys())

# reshape into array
DV = np.reshape(DVs,(idx+1,n_dv),order = 'C')

# iterations
iterations = np.arange(idx+1)+1

# initialize plot
fig1,ax1 = plt.subplots(1,1)
ax1.set_ylabel('Ballast Volume [m**3]',fontsize = 16)
ax1.set_xlabel('Iterations',fontsize = 16)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

# get the name of design variables
leg_names = list(dvs.keys())

# plot each design variable 
for i in range(n_dv):
    ax1.plot(iterations,DV[:,i],'-*',label = leg_names[i])
    
# plot LCOE  
ax2 = ax1.twinx()
ax2.plot(iterations,LCOE,'o-',color = 'black',label = 'LCOE')
ax2.set_ylabel('LCOE [$/MWh]',fontsize = 16)
ax2.tick_params(axis='y', labelsize=12)


fig1.legend( loc='upper center',ncol = 4,fontsize = 12) 
