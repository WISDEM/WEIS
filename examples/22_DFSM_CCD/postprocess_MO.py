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
    cs = [42,51,57,65]
    npts_list = [10,10,10,9]

    dir_names = ['/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_42_4var2', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_51_4var2', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_57_4var2', '/home/athulsun/WEIS-AKS-DEV/examples/20_MultiFid/outputs/rated_65_4var2'] 

    n_cs = len(cs)

    DEL = []
    GSstd = []

    save_dir = OF_dir + os.sep + 'MO_CCD_4var'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    obj1 = 'TwrBsMyt_DEL'
    obj2 = 'GenSpeed_Std'


    fig,ax = plt.subplots()
    ax.set_xlabel(obj1,fontsize = fontsize_axlabel)
    ax.set_ylabel(obj2,fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.grid()

    pareto_points_all = []
    DV = []
    wt_inds = []


    for i_cs in range(n_cs):

        n_pts = npts_list[i_cs]

        objs = np.zeros((n_pts,2))
        pareto_points = np.zeros((n_pts,2))
        dv_opt = np.zeros((n_pts,4))

        w1 = np.linspace(1,0,n_pts)
        w2 = 1-w1

        dir_n = dir_names[i_cs]

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

        dv_opt = dv_opt[non_dom_indices]
        wt = wt[non_dom_indices]


        sort_ind = np.argsort(non_dom_points[:,0])

        non_dom_points = non_dom_points[sort_ind]
        dv_opt = dv_opt[sort_ind]
        wt = wt[sort_ind]

        DEL.append(np.min(non_dom_points[:,0]))
        GSstd.append(np.min(non_dom_points[:,1]))


        pareto_points_all.append(non_dom_points)
        DV.append(dv_opt)
        wt_inds.append(wt)

        
        ax.plot(non_dom_points[:,0],non_dom_points[:,1],'-o',markersize = markersize,label = 'CS = '+str(cs[i_cs]))
        


        if cs[i_cs] == 51:
            f,a = plt.subplots(1)
            a.plot(non_dom_points[:,0],non_dom_points[:,1],'-o',markersize = markersize*2,color = '#2ca02c')
            a.set_xlabel(obj1,fontsize = fontsize_axlabel)
            a.set_ylabel(obj2,fontsize = fontsize_axlabel)
            a.tick_params(labelsize=fontsize_tick)
            a.grid()
        
    
    pareto_points_all = np.vstack(pareto_points_all)
    non_dom_ind = find_nondominated(pareto_points_all)
    pareto_points_all = pareto_points_all[non_dom_ind]

    ax.plot(pareto_points_all[:,0],pareto_points_all[:,1],'.k',label = 'Pareto-front')
    

    a.plot(pareto_points_all[-2:,0],pareto_points_all[-2:,1],'-o',markersize = markersize*2,color = '#e377c2')
    f.savefig('pareto-pareto.pdf')

    ax.legend(ncol = 3,fontsize = fontsize_legend)
    fig.savefig(save_dir+os.sep+'Pareto_CCD.pdf')


    f1,a1 = plt.subplots(1)
    a1.plot(cs,DEL,'k--',markersize = markersize*2)
    for i in range(n_cs):
        a1.plot(cs[i],DEL[i],'o',markersize = markersize*2)
    
    a1.set_xlabel('Column Spacing [m]',fontsize = fontsize_axlabel)
    a1.set_ylabel(obj1,fontsize = fontsize_axlabel)
    a1.tick_params(labelsize=fontsize_tick)
    f1.savefig(save_dir+os.sep+'DELvsCS.pdf')

    f1,a1 = plt.subplots(1)
    a1.plot(cs,GSstd,'k--',markersize = markersize*2)
    for i in range(n_cs):
        a1.plot(cs[i],GSstd[i],'o',markersize = markersize*2)
    a1.set_xlabel('Column Spacing [m]',fontsize = fontsize_axlabel)
    a1.set_ylabel(obj2,fontsize = fontsize_axlabel)
    a1.tick_params(labelsize=fontsize_tick)
    f1.savefig(save_dir+os.sep+'GSstdvsCS.pdf')

    plt.show()

    breakpoint()
    




