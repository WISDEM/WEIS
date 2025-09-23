import numpy as np
import matplotlib.pyplot as plt
import dill
import os

# plot properties
markersize = 5
linewidth = 1.5
fontsize_legend = 12
fontsize_axlabel = 18
fontsize_tick = 15
format = '.pdf'

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def is_dominated(p, q):
    """Returns True if point p is dominated by point q (to be minimized)."""
    return np.all(q <= p) and np.any(q < p)

def find_nondominated(points):
    """Returns the indices of non-dominated points."""
    num_points = len(points)
    nondominated = []
    for i in range(num_points):
        dominated = False
        for j in range(num_points):
            if i != j and is_dominated(points[i], points[j]):
                dominated = True
                break
        if not dominated:
            nondominated.append(i)
    return nondominated


if __name__ == '__main__':

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs'
    cs = [35,42,51,57,65]
    npts_list = [10,10,10,10,9]

    dir_names1 = ['/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_35_4var','/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_42_4var2', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_51_4var2', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_57_4var2', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_65_4var'] #[OF_dir + os.sep + 'rated_'+str(i)+'_4var2' for i in cs]
    dir_names2 = ['/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_35_4var','/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_42_4var', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_51_4var', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_57_4var', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_65_4var2'] #[OF_dir + os.sep + 'rated_'+str(i)+'_4var2' for i in cs]

    dir_names = np.array([dir_names1,dir_names2])
    dir_names = dir_names.T

    n_pts_list = np.array([[10,10],
                      [10,10],
                      [10,10],
                      [10,10],
                      [10,9]])
    
    
    
    n_cs = len(cs)

    DEL = []
    GSstd = []

    save_dir = OF_dir + os.sep + 'CCD_pareto'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    obj1 = 'TwrBsMyt_DEL'
    obj2 = 'GenSpeed_Std'

    pareto_all = []

    fig,ax = plt.subplots(1)
    ax.set_xlabel(obj1,fontsize = fontsize_axlabel)
    ax.set_ylabel(obj2,fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.grid()

    for i_cs in range(n_cs):

        pareto_per_fol = []

        for i_fols in range(2):

            fol1 = dir_names[i_cs,i_fols]

            
            n_pts = n_pts_list[i_cs,i_fols]

            objs = np.zeros((n_pts,2))
            pareto_points = np.zeros((n_pts,2))
            dv_opt = np.zeros((n_pts,4))

            w1 = np.linspace(1,0,n_pts)
            w2 = 1-w1

            dir_n = dir_names[i_cs,i_fols]

            wt = np.linspace(1,0,n_pts)

            for i_pt in range(n_pts):

            
                hf_warmstart_file_iter = dir_n+os.sep+'multiobj_iter_hf_'+ str(i_pt)+'.dill'

                with open(hf_warmstart_file_iter,'rb') as handle:
                    hf_results_iter = dill.load(handle)

                desvars = np.array(hf_results_iter['desvars'])
                n_iter = len(desvars)

                outputs_hf_iter = np.zeros((n_iter,2))
                w1_ = w1[i_pt]
                w2_ = w2[i_pt] 

                for i in range(n_iter):
                    outputs_hf_iter[i,0] = hf_results_iter['outputs'][i][obj1]
                    outputs_hf_iter[i,1] = hf_results_iter['outputs'][i][obj2]

                pareto_points[i_pt,0] = outputs_hf_iter[-1,0]
                pareto_points[i_pt,1] = outputs_hf_iter[-1,1]
                dv_opt[i_pt,:] = desvars[-1,:]

            non_dom_indices = find_nondominated(pareto_points)
            non_dom_points = pareto_points[non_dom_indices]

            pareto_per_fol.append(non_dom_points)

            if i_cs == 2:
                breakpoint()
        
        #diff_pareto = pareto_per_fol[0] - pareto_per_fol[1]

        if i_cs in [35,65]:
            
            pareto_per_cs = pareto_per_fol[0]
            pareto_per_cs = pareto_per_cs[sort_ind,:]
            pareto_per_cs.append(pareto_per_cs)

            
        else:

            pareto_per_fol = np.vstack(pareto_per_fol)

            non_dom_indices = find_nondominated(pareto_per_fol)
            pareto_per_cs = pareto_per_fol[non_dom_indices]

            sort_ind = np.argsort(pareto_per_cs[:,0])

            pareto_per_cs = pareto_per_cs[sort_ind,:]

        pareto_all.append(pareto_per_cs)
        DEL.append(np.min(pareto_per_cs[:,0]))
        GSstd.append(np.min(pareto_per_cs[:,1]))

        ax.plot(pareto_per_cs[:,0],pareto_per_cs[:,1],'-o',markersize = markersize)

    pareto_all = np.vstack(pareto_all)
    non_dom_indices = find_nondominated(pareto_all)
    pareto_all = pareto_all[non_dom_indices]

    ax.plot(pareto_all[:,0],pareto_all[:,1],'.k')
    fig.savefig(save_dir + os.sep +'pareto_all.pdf')

    #---------------------------------------------------------

    fig,ax = plt.subplots(1)
    ax.set_xlabel(obj1,fontsize = fontsize_axlabel)
    ax.set_ylabel(obj2,fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.grid()

    ax.plot(pareto_all[:7,0],pareto_all[:7,1],'-o',markersize = markersize,color = '#2ca02c')
    ax.plot(pareto_all[7,0],pareto_all[7,1],'-o',markersize = markersize,color = '#d62728')
    ax.plot(pareto_all[8:,0],pareto_all[8:,1],'-o',markersize = markersize,color = '#9467bd')
    fig.savefig(save_dir + os.sep +'pareto_focus.pdf')

    #------------------------------------------------------------
    fig,ax = plt.subplots(1)
    ax.set_xlabel('Column Spacing [m]',fontsize = fontsize_axlabel)
    ax.set_ylabel(obj1,fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.grid()

    ax.plot(cs,DEL,'--',markersize = markersize-3,color = 'k')

    for i_cs in range(n_cs):
        ax.plot(cs[i_cs],DEL[i_cs],'-o',markersize = markersize+5)
    fig.savefig(save_dir + os.sep +'DEL.pdf')

    #------------------------------------------------------------
    fig,ax = plt.subplots(1)
    ax.set_xlabel('Column Spacing [m]',fontsize = fontsize_axlabel)
    ax.set_ylabel(obj2,fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.grid()

    ax.plot(cs,GSstd,'--',markersize = markersize-3,color = 'k')
    for i_cs in range(n_cs):
        ax.plot(cs[i_cs],GSstd[i_cs],'-o',markersize = markersize+5)

    fig.savefig(save_dir + os.sep +'DELopt.pdf')

    plt.show()
    breakpoint()







