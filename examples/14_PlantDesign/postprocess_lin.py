import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt


mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
opt_path = mydir + os.sep + "outputs" + os.sep +"BaDi1_COBYLA_single"
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

for idx, case in enumerate(driver_cases[:-1]):
    #print('===================')
    #print('Simulation index:', ABCD_list[idx]['sim_idx'])
    dvs = case.get_design_vars(scaled=False)
    for key in dvs.keys():
       # print(key)
        #print(dvs[key])
        DVs.append(dvs[key])
    A_plot.append(ABCD_list[idx]['x_ops'][0])
    #print()
    #print("A matrix")
    #print(ABCD_list[idx]['A'])
    #print()
    
    #A_plot.append(ABCD_list[idx]['A'][1, 1])

ind = ABCD_list[0]['DescOutput'].index('ED TwrBsFxt, (kN)')

DV = np.reshape(DVs,(idx+1,4),order = 'C')

A_plot = np.rad2deg(A_plot)

PP = np.hstack([DV,A_plot])

fig,ax = plt.subplots(1,1)
ax.plot(PP[:,0],PP[:,1],'*-')
ax.set_xlabel('Column 3 BV [m**3]',fontsize = 16)
ax.set_ylabel('PtfmPitch [deg]',fontsize = 16)

iterations = np.arange(idx+1)+1

fig1,ax1 = plt.subplots(1,1)
ax1.set_ylabel('Ballast Volume [m**3]',fontsize = 16)
ax1.set_xlabel('Iterations',fontsize = 16)
#ax1.set_title('Ballast Volume vs Iterations',fontsize = 12)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
#ax1.set_xticks(iterations)

leg_names = ['Main Column','Column1','Column2','Column3']

for i in range(4):
    ax1.plot(iterations,DV[:,i],'-*',label = leg_names[i])

    
fig1.legend( loc='upper center',ncol = 4,fontsize = 12) 
# n_sample = len(driver_cases); ndvs = len(dvs)

# DV = np.reshape(DVs,(n_sample,ndvs),order = 'C')

# E = DV[:,0]; rho = DV[:,1]

# Ex,Rx = np.meshgrid(E,rho)

# rho = np.floor(rho)
# fig, ax = plt.subplots(1,1,)
# #fig, ax2 = plt.subplots(4,1,)
# for i in range(len(ABCD_list)):
#     u_h = ABCD_list[i]["u_h"]
#     u_ops = ABCD_list[i]["u_ops"]
#     x_ops = ABCD_list[i]["x_ops"]
    
    
#     plt.plot(u_h,(x_ops[ind,:]),label = "BV ={:} ".format(DVs[i]) )
#     plt.ylabel('PtfmPitch [rad]')
#     plt.xlabel('Wind Speed [m/s]')

#plt.legend(loc = "upper right")

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