import numpy as np
import matplotlib.pyplot as plt
import dill
import os
import pickle

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
    n_pts = 10

    results_folder = OF_dir + os.sep + 'multi_fid_results' 
    results_file = results_folder+ os.sep + 'MO_results.pkl'

    # load results
    with open(results_file,'rb') as handle:
        MO_results = pickle.load(handle)

    # extract results
    obj1 = MO_results['obj1']
    obj2 = MO_results['obj2']
    objs = MO_results['objs']
    opt_pts = MO_results['opt_pts']


    # initialize plots
    fig,ax = plt.subplots()
    ax.set_xlabel(obj1,fontsize = fontsize_axlabel)
    ax.set_ylabel(obj2,fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.grid()

    # find indices of non dominated pooints
    non_dom_ind = find_nondominated(objs)
    pareto_points = objs[non_dom_ind,:]

    # plot nondominated points
    fig,ax.plot(objs[:,0],objs[:,1],'o',markersize = 8)

    # save figure
    fig.savefig(results_folder + os.sep + 'pareto-points.pdf')

    plt.show()





