import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 15
format = '.pdf'

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

'''
File to plot contours of key performance metrics

'''


if __name__ == '__main__':

    # get path to this dir
    this_dir = os.path.dirname(os.path.realpath(__file__))

    results_file = this_dir + os.sep +'outputs/sensstudy_results_OZ.pkl'
    plot_path = this_dir + os.sep +'outputs/sens_results_OZ'

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    with open(results_file,'rb') as handle:
        results_dict = pickle.load(handle)

    OZ = results_dict['OZ']

    omega_pc = np.unique(OZ[:,0])
    zeta_pc = np.unique(OZ[:,1])

    O,Z = np.meshgrid(omega_pc,zeta_pc)

    samples_shape = np.shape(O)


    key_metrics = [metric for metric in results_dict.keys()]

    for metric in key_metrics:

        if metric == 'OZ':
            continue
        else:

            metric_of = results_dict[metric][:,0]
            metric_dfsm = results_dict[metric][:,1]

            if metric == 'twrbsmyt_del':
                metric_name = 'TwrBsMyt_DEL [MNm]'
                metric_of*=1e2
                metric_dfsm*=1e2
            elif metric == 'genspeed_std':
                metric_name = 'GenSpeed_Std'
            else:
                metric_name = metric

            metric_of = np.reshape(metric_of,samples_shape,order = 'F')
            metric_dfsm = np.reshape(metric_dfsm,samples_shape,order = 'F')

            fig,ax = plt.subplots(1)

            ax.set_xlabel('Omega PC',fontsize = fontsize_axlabel)
            ax.set_ylabel('Zeta PC',fontsize = fontsize_axlabel)
            CP = ax.contourf(O,Z,metric_of,25,cmap = plt.cm.viridis)
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_title('OpenFAST',fontsize = fontsize_axlabel)

            cbar = fig.colorbar(CP)
            cbar.set_label(metric_name,fontsize = fontsize_axlabel)

            fig.savefig(plot_path +os.sep +metric + '_OF'+ format)
            plt.close(fig)

            fig,ax = plt.subplots(1)

            ax.set_xlabel('Omega PC',fontsize = fontsize_axlabel)
            ax.set_ylabel('Zeta PC',fontsize = fontsize_axlabel)
            CP = ax.contourf(O,Z,metric_dfsm,15,cmap = plt.cm.viridis)
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_title('DFSM',fontsize = fontsize_axlabel)

            cbar = fig.colorbar(CP)
            cbar.set_label(metric_name,fontsize = fontsize_axlabel)

            fig.savefig(plot_path +os.sep +metric + '_DFSM'+ format)
            plt.close(fig)

    