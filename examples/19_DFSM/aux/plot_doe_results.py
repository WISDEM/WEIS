import numpy as np
import pickle
import os 
import matplotlib.pyplot as plt

if __name__ == '__main__':

    this_dir = os.path.dirname(os.path.realpath(__file__))

    results_file = 'doe_comparison_results.pkl'
    plot_path = 'Q3_results'

    with open(results_file,'rb') as handle:
        results = pickle.load(handle)

    # extract results
    omega_pc_list = results['omega_pc_list']
    genspeed_max_of = results['genspeed_max_of']
    genspeed_max_dfsm = results['genspeed_max_dfsm']
    ptfmpitch_max_of = results['ptfmpitch_max_of']
    ptfmpitch_max_dfsm = results['ptfmpitch_max_dfsm']
    DEL_of = results['DEL_of']
    DEL_dfsm = results['DEL_dfsm']


    save_flag = True
    col_error = 'k'

    # plot properties
    w = 7.5;h = 5
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 12+3
    fontsize_tick = 12-2

    fig,ax = plt.subplots(1)
    fig.set_size_inches(w,h)

    ax2 = ax.twinx()
    ax2.plot(omega_pc_list,np.abs((genspeed_max_of - genspeed_max_dfsm)/genspeed_max_of)*100,'.--',label = 'error',markersize = markersize,color = col_error,alpha = 0.7)
    ax2.set_ylabel('Percentage Error [%]',fontsize = fontsize_axlabel,color = col_error)
    ax2.tick_params(labelsize=fontsize_tick,labelcolor = col_error)


    ax.plot(omega_pc_list,genspeed_max_of,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(omega_pc_list,genspeed_max_dfsm,'.-',label = 'DFSM',markersize = markersize)

    

    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel)
    ax.set_ylabel('Max. GenSpeed [rpm]',fontsize = fontsize_axlabel)
    #ax.legend(ncol = 3,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'genspeed_max.pdf')


    fig,ax = plt.subplots(1)
    fig.set_size_inches(w,h)

    ax2 = ax.twinx()
    ax2.plot(omega_pc_list,np.abs((ptfmpitch_max_of - ptfmpitch_max_dfsm)/ptfmpitch_max_of)*100,'.--',label = 'error',markersize = markersize,color = col_error,alpha = 0.7)
    ax2.set_ylabel('Percentage Error [%]',fontsize = fontsize_axlabel,color = col_error)
    ax2.tick_params(labelsize=fontsize_tick,labelcolor = col_error)


    ax.plot(omega_pc_list,ptfmpitch_max_of,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(omega_pc_list,ptfmpitch_max_dfsm,'.-',label = 'DFSM',markersize = markersize)
    #ax.plot(omega_pc_list,ptfmpitch_max_of - ptfmpitch_max_dfsm,'.-',label = 'error',markersize = markersize,color = 'k')

    

    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel)
    ax.set_ylabel('Max. PtfmPitch [deg]',fontsize = fontsize_axlabel)
    #ax.legend(ncol = 3,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'ptfmpitch_max.pdf')


    fig,ax = plt.subplots(1)
    fig.set_size_inches(w,h)

    ax2 = ax.twinx()
    ax2.plot(omega_pc_list,np.abs((DEL_of - DEL_dfsm)/DEL_of)*100,'.--',label = 'error',markersize = markersize,color = col_error,alpha = 0.7)
    ax2.set_ylabel('Percentage Error [%]',fontsize = fontsize_axlabel,color = col_error)
    ax2.tick_params(labelsize=fontsize_tick,labelcolor = col_error)


    ax.plot(omega_pc_list,DEL_of,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(omega_pc_list,DEL_dfsm,'.-',label = 'DFSM',markersize = markersize)
   
    
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel)
    ax.set_ylabel('DEL_twrbsmyt [kNm]',fontsize = fontsize_axlabel)
    #ax.legend(ncol = 3,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'DEL.pdf')

    # create a figure with one subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_name = 'legend'
    ax.plot([1,2,3,4,5], [1,2,3,4,5], '.-',label='OpenFAST')
    ax.plot([1,2,3,4,5], [1,2,3,4,5], '.-',label='DFSM')
    ax.plot([1,2,3,4,5], [1,2,3,4,5],'.--' ,label='Pct. Error',color = col_error,alpha = 0.7)
    
    # then create a new image
    # adjust the figure size as necessary
    figsize = (4, 0.5)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center',ncol = 3)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    
    if save_flag:
        fig_leg.savefig(plot_path +os.sep + plot_name + '.pdf')


    plt.show()
