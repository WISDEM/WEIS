import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 15

if __name__ == '__main__':

    dfsm_results_file = 'closed_loop_validation_test_DFSM2/DFSM_validation_results.pkl'
    n4sid_results_file = 'sys_id_test/CL_validation/results_dict.pkl'
    lstm_results_file = 'RNN_models/LSTM_14_test/plots_validations/results_dict.pkl'

    with open(dfsm_results_file,'rb') as handle:
        dfsm_results = pickle.load(handle)

    with open(n4sid_results_file,'rb') as handle:
        n4sid_results = pickle.load(handle)

    with open(lstm_results_file,'rb') as handle:
        lstm_results = pickle.load(handle)

    #--------------------------------------
    fig,ax = plt.subplots(1)

    ax.semilogx(np.mean(n4sid_results['model_sim_time']),np.mean(n4sid_results['bp_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['bp_mse'],'.',color = 'tab:orange',label = 'n4sid',markersize = markersize)
    
    ax.semilogx(np.mean(dfsm_results['model_sim_time']),np.mean(dfsm_results['bp_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['bp_mse'],'.',color = 'r',label = 'DFSM',markersize = markersize)
    

    ax.semilogx(np.mean(lstm_results['model_sim_time']),np.mean(lstm_results['bp_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(lstm_results['model_sim_time'],lstm_results['bp_mse'],'.',color = 'b',label = 'LSTM',markersize = markersize)
    


    ax.set_xticks([1, 10, 20,30,40,50,60,70,80,90,100])
    ax.grid(alpha = 0.5)

    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 1,fontsize = fontsize_legend)
    ax.set_title('MSE Blade Pitch',fontsize = fontsize_axlabel)
    ax.set_xlabel('Simulation Time [s]',fontsize = fontsize_axlabel)

    plot_name = 'BP_mse.pdf'

    fig.savefig(plot_name)
    #--------------------------------------

    fig,ax = plt.subplots(1)
    
    ax.semilogx(np.mean(n4sid_results['model_sim_time']),np.mean(n4sid_results['twrbsmyt_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['twrbsmyt_mse'],'.',color = 'tab:orange',label = 'n4sid',markersize = markersize)
    
    ax.semilogx(np.mean(dfsm_results['model_sim_time']),np.mean(dfsm_results['twrbsmyt_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['twrbsmyt_mse'],'.',color = 'r',label = 'DFSM',markersize = markersize)
    

    ax.semilogx(np.mean(lstm_results['model_sim_time']),np.mean(lstm_results['twrbsmyt_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(lstm_results['model_sim_time'],lstm_results['twrbsmyt_mse'],'.',color = 'b',label = 'LSTM',markersize = markersize)

    ax.set_xticks([1, 10, 20,30,40,50,60,70,80,90,100])
    ax.grid(alpha = 0.5)


    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 1,fontsize = fontsize_legend)
    ax.set_title('MSE Twr. Base Moment',fontsize = fontsize_axlabel)
    ax.set_xlabel('Simulation Time [s]',fontsize = fontsize_axlabel)

    plot_name = 'My_mse.pdf'

    fig.savefig(plot_name)
    #--------------------------------------
    fig,ax = plt.subplots(1)
    
    ax.semilogx(np.mean(n4sid_results['model_sim_time']),np.mean(n4sid_results['gs_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['gs_mse'],'.',color = 'tab:orange',label = 'n4sid',markersize = markersize)
    
    ax.semilogx(np.mean(dfsm_results['model_sim_time']),np.mean(dfsm_results['gs_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['gs_mse'],'.',color = 'r',label = 'DFSM',markersize = markersize)
    

    ax.semilogx(np.mean(lstm_results['model_sim_time']),np.mean(lstm_results['gs_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(lstm_results['model_sim_time'],lstm_results['gs_mse'],'.',color = 'b',label = 'LSTM',markersize = markersize)

    ax.set_xticks([1, 10, 20,30,40,50,60,70,80,90,100])
    ax.grid(alpha = 0.5)


    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 1,fontsize = fontsize_legend)
    ax.set_title('MSE Generator Speed',fontsize = fontsize_axlabel)
    ax.set_xlabel('Simulation Time [s]',fontsize = fontsize_axlabel)

    plot_name = 'GS_mse.pdf'

    fig.savefig(plot_name)

    #--------------------------------------
    fig,ax = plt.subplots(1)
    
    ax.semilogx(np.mean(n4sid_results['model_sim_time']),np.mean(n4sid_results['ps_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['ps_mse'],'.',color = 'tab:orange',label = 'n4sid',markersize = markersize)
    
    ax.semilogx(np.mean(dfsm_results['model_sim_time']),np.mean(dfsm_results['ps_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['ps_mse'],'.',color = 'r',label = 'DFSM',markersize = markersize)
    

    ax.semilogx(np.mean(lstm_results['model_sim_time']),np.mean(lstm_results['ps_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(lstm_results['model_sim_time'],lstm_results['ps_mse'],'.',color = 'b',label = 'LSTM',markersize = markersize)

    ax.set_xticks([1, 10, 20,30,40,50,60,70,80,90,100])
    ax.grid(alpha = 0.5)


    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 1,fontsize = fontsize_legend)
    ax.set_title('MSE Platform Surge',fontsize = fontsize_axlabel)
    ax.set_xlabel('Simulation Time [s]',fontsize = fontsize_axlabel)

    plot_name = 'PS_mse.pdf'

    fig.savefig(plot_name)

    #--------------------------------------
    fig,ax = plt.subplots(1)
    
    ax.semilogx(np.mean(n4sid_results['model_sim_time']),np.mean(n4sid_results['pp_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['pp_mse'],'.',color = 'tab:orange',label = 'n4sid',markersize = markersize)
    
    ax.semilogx(np.mean(dfsm_results['model_sim_time']),np.mean(dfsm_results['pp_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['pp_mse'],'.',color = 'r',label = 'DFSM',markersize = markersize)
    

    ax.semilogx(np.mean(lstm_results['model_sim_time']),np.mean(lstm_results['pp_mse']),'*',color = 'k',markersize = markersize+6)
    ax.semilogx(lstm_results['model_sim_time'],lstm_results['pp_mse'],'.',color = 'b',label = 'LSTM',markersize = markersize)

    ax.set_xticks([1, 10, 20,30,40,50,60,70,80,90,100])
    ax.grid(alpha = 0.5)

    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 1,fontsize = fontsize_legend)
    ax.set_title('MSE Platform Pitch',fontsize = fontsize_axlabel)
    ax.set_xlabel('Simulation Time [s]',fontsize = fontsize_axlabel)

    plot_name = 'PP_mse.pdf'

    fig.savefig(plot_name)

    #--------------------------------------

    qty_lists = ['bp_mse','gs_mse','twrbsmyt_mse']
    title_list = ['MSE Blade Pitch','MSE Generator Speed','MSE Tower-Base Moment']

    for i,qty in enumerate(qty_lists):

        fig,ax = plt.subplots(1)

        mse = [n4sid_results[qty],dfsm_results[qty],lstm_results[qty]]
        colors = ['tab:orange','r','b']
        labels = ['n4sid','DFSM','LSTM']

        bplot = ax.boxplot(mse,patch_artist=True,tick_labels=labels, medianprops={
        "color": "black",         # line color
        "linestyle": "-",         # optional: solid line
        "linewidth": 1.5 })

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        ax.tick_params(labelsize=fontsize_tick)
        ax.set_title(title_list[i],fontsize = fontsize_axlabel)
        plot_name = qty+'_box.pdf'
        fig.savefig(plot_name)



