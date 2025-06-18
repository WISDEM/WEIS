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

    dfsm_results_file = '/home/athulsun/DEV/examples/19_DFSM/closed_loop_validation_test_DFSM/results_dict.pkl'
    n4sid_results_file = '/home/athulsun/DEV/examples/19_DFSM/sys_id_test/CL_validation/results_dict.pkl'
    lstm_results_file = '/home/athulsun/DEV/examples/19_DFSM/RNN_models/LSTM_14/plots_validations/results_dict.pkl'

    with open(dfsm_results_file,'rb') as handle:
        dfsm_results = pickle.load(handle)

    with open(n4sid_results_file,'rb') as handle:
        n4sid_results = pickle.load(handle)

    with open(lstm_results_file,'rb') as handle:
        lstm_results = pickle.load(handle)

    fig,ax = plt.subplots(1)

    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['bp_mse'],'.',color = 'tab:orange',label = 'n4sid',markersize = markersize)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['bp_mse'],'.',color = 'r',label = 'DFSM',markersize = markersize)
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

    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['twrbsmyt_mse'],'.',label = 'n4sid',markersize = markersize)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['twrbsmyt_mse'],'.',label = 'DFSM',markersize = markersize)
    ax.semilogx(lstm_results['model_sim_time'],lstm_results['twrbsmyt_mse'],'.',label = 'LSTM',markersize = markersize)
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

    ax.semilogx(n4sid_results['model_sim_time'],n4sid_results['gs_mse'],'.',label = 'n4sid',markersize = markersize)
    ax.semilogx(dfsm_results['model_sim_time'],dfsm_results['gs_mse'],'.',label = 'DFSM',markersize = markersize)
    ax.semilogx(lstm_results['model_sim_time'],lstm_results['gs_mse'],'.',label = 'LSTM',markersize = markersize)
    ax.set_xticks([1, 10, 20,30,40,50,60,70,80,90,100])
    ax.grid(alpha = 0.5)


    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 1,fontsize = fontsize_legend)
    ax.set_ylabel('MSE Gen. Speed',fontsize = fontsize_axlabel)
    ax.set_xlabel('Simulation Time [s]',fontsize = fontsize_axlabel)

    plot_name = 'GS_mse.pdf'

    fig.savefig(plot_name)

    breakpoint()
    