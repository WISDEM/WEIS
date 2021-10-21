import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
opt_path = mydir + os.sep + "outputs" + os.sep + "tower_doe_run" 
pkl_path = opt_path + os.sep+  "ABCD_matrices.pkl"

with open(pkl_path, 'rb') as handle:

    ABCD_list = pickle.load(handle)
  
print("Information available in the pickle file:")
for key in ABCD_list[0]:
    print(key)
print()    


cr = om.CaseReader(opt_path+ os.sep +"log_opt.sql")

driver_cases = cr.get_cases('driver')

A_plot = []
DVs = []
for idx, case in enumerate(driver_cases):
    print('===================')
    print('Simulation index:', ABCD_list[idx]['sim_idx'])
    dvs = case.get_design_vars(scaled=False)
    for key in dvs.keys():
        print(key)
        print(dvs[key])
        DVs.append(dvs[key])
    print()
    print("A matrix")
    #print(ABCD_list[idx]['A'])
    print()
    
    A_plot.append(ABCD_list[idx]['A'][1, 1])
    
rho = np.array([DVs[0][0],DVs[2][0],DVs[4][0],DVs[6][0]])

rho = np.floor(rho)
fig, ax = plt.subplots(1,1,)
#fig, ax2 = plt.subplots(4,1,)
for i in range(len(ABCD_list)):
    u_h = ABCD_list[i]["u_h"]
    u_ops = ABCD_list[i]["u_ops"]
    x_ops = ABCD_list[i]["x_ops"]
    
    
    plt.plot(u_h,np.rad2deg(x_ops[0,:]),label = "rho ={:} ".format(rho[i]) )
    plt.ylabel('Ptfm Pitch [deg]')
    plt.xlabel('Wind Speed [m/s]')

plt.legend(loc = "upper right")
# 
# 
# A_plot = np.array(A_plot)
# DVs = np.array(DVs)
# 
# plt.scatter(DVs, A_plot[:])
# 
# plt.xlabel("Tower Young's Modulus, Pa")
# plt.ylabel('A[1, 1]')
# plt.tight_layout()
# 
# plt.show()