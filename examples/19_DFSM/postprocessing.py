import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

def extract_results(driver_cases,obj,const = None):

    DV = [];Obj = [];Con = []

    for idx, case in enumerate(driver_cases):

        dvs = case.get_design_vars(scaled=False)

        Obj.append(case.get_objectives(scaled = False)[obj][0])

        if not(const == None):
            Con.append(case.get_constraints(scaled = False)[const][0])

        for key in dvs.keys():
            DV.append(dvs[key])

    n_iter = idx+1
    n_var = len(dvs.keys())

    Obj = np.array(Obj)
    DV = np.array(DV)

    if not(const == None):
        Con = np.array(Con)


    return Obj,DV,Con,n_iter,n_var




this_dir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

fols = ['RM1_test_3']


n_tests = len(fols)

fig,axs = plt.subplots(n_tests,1, constrained_layout=True)
fig2,axs2 = plt.subplots(n_tests,1, constrained_layout=True)
levels = np.arange(500,5500,500)

for ifol,fol in enumerate(fols):

    ax = axs
    ax2 = axs2

    doe_output_dir = this_dir + os.sep + 'outputs' + os.sep + fol
    doe_sql_file = doe_output_dir + os.sep + 'log_opt.sql'

    obj = 'aeroelastic.DEL_TwrBsMyt'
    con = 'aeroelastic.rotor_overspeed'
    n_samples = 6

    # load sql file
    cr_doe = om.CaseReader(doe_sql_file)
    doe_driver_cases = cr_doe.get_cases('driver')


    # load design variables
    DEL,DV,Con,n_iter,n_var = extract_results(doe_driver_cases,obj,con)
    DEL_low = np.argmin(DEL)

    DEL = np.reshape(DEL,[n_samples,n_samples])
    Con = np.reshape(Con,[n_samples,n_samples])
    DV = np.reshape(DV,[n_iter,n_var],order = 'C')
    DV_opt = DV[DEL_low,:]
    
    
    omega_pc = np.unique(DV[:,0])
    zeta_pc = np.unique(DV[:,1])

    O,Z = np.meshgrid(omega_pc,zeta_pc)

    #ig.suptitle(title)
    CP = ax.contourf(O,Z,DEL,25,cmap = plt.cm.viridis)
    #ax.set_title(title[ifol])
    ax.set_xlabel('Omega PC')
    ax.plot(DV_opt[0],DV_opt[1],'.',color = 'r',markersize = 10)
    #ax.set_xlim([0.1,1])
    

    CP2 = ax2.contourf(O,Z,Con,25,cmap = plt.cm.viridis)
    ax2.contour(O,Z,Con,levels = [0.25],colors = 'r',linewidth = 2)
    #ax2.set_title(title[ifol])
    ax2.set_xlabel('Omega PC')
    
    #ax2.set_xlim([0.1,1])



fig2.suptitle('RO <= 0.22')
axs.set_ylabel('Zeta PC')
axs2.set_ylabel('Zeta PC')
cbar = fig.colorbar(CP)
cbar.ax.set_ylabel('DEL')

cbar2 = fig.colorbar(CP2)
cbar2.ax.set_ylabel('Rotor Overspeed')

# plot_name = 'Kp_omg_6.png'

# fig.savefig(plot_name)



# scats = []

# def animate(i):
#     global scats

#     for scat in scats:
#         scat.remove()
#     scats = []

#     if i >= n_iter_opt-1:
#         scat = ax.scatter(DV_opt[-1,0],DV_opt[-1,1],c = 'tab:green',s = 40)

#         title = 'Iteration {:02d}'.format(n_iter_opt)

#     else:

#         scat = ax.scatter(DV_opt[i,0],DV_opt[i,1],c = 'r',s = 40)

#         title = 'Iteration {:02d}'.format(i+1)


#     # set picture
#     ax.set_title(title)

#     scats.append(scat)

#     return ax

# # create animation and save
# savename = 'optimization3' +  '.gif'
# anim = FuncAnimation(fig,animate,frames = n_iter_opt+200,interval=500,**{'repeat' : False})
# anim.save(savename, dpi=120, writer="pillow")


plt.show()
# breakpoint()