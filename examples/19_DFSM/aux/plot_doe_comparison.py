import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()

    return 1 - (ss_res/ss_tot)


if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # pkl files
    doe_results_lstm_pkl = 'doe_comparison_results_lstm.pkl' 
    doe_results_n4sid_pkl = 'doe_comparison_results_ssid.pkl' 

    # load pickle files
    with open(doe_results_lstm_pkl,'rb') as handle:
        results_lstm = pickle.load(handle)

    with open(doe_results_n4sid_pkl,'rb') as handle:
        results_n4sid = pickle.load(handle)

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 12
    fontsize_axlabel = 15
    fontsize_tick = 12

    plot_path = 'doe_comp'
    save_flag = True

    
    omega_pc_list = results_lstm['omega_pc_list']
    genspeed_max_of = results_lstm['genspeed_max_of']
    genspeed_max_lstm = results_lstm['genspeed_max_dfsm']
    genspeed_max_n4sid = results_n4sid['genspeed_max_dfsm']

    r2_gs_of = np.corrcoef(omega_pc_list,genspeed_max_of)[0,1]
    r2_gs_n4sid = np.corrcoef(omega_pc_list,genspeed_max_n4sid)[0,1]
    r2_gs_lstm = np.corrcoef(omega_pc_list,genspeed_max_lstm)[0,1]

    fig,ax = plt.subplots(1)

    ax.plot(omega_pc_list,genspeed_max_of,'.-',markersize = markersize,linewidth = linewidth,label = 'OpenFAST')
    ax.plot(omega_pc_list,genspeed_max_n4sid,'.-',markersize = markersize,linewidth = linewidth,label = 'n4sid')
    ax.plot(omega_pc_list,genspeed_max_lstm,'.-',markersize = markersize,linewidth = linewidth,label = 'LSTM')


    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel)
    ax.set_ylabel('Max. GenSpeed [rpm]',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 3,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        fig.savefig(plot_path +os.sep+ 'GS_max_comp.pdf')
    plt.close(fig)

    # DEL
    del_of = results_lstm['DEL_of']
    del_lstm = results_lstm['DEL_dfsm']
    del_n4sid = results_n4sid['DEL_dfsm']
    
    r2_del_of = np.corrcoef(omega_pc_list,del_of)[0,1]
    r2_del_n4sid = np.corrcoef(omega_pc_list,del_n4sid)[0,1]
    r2_del_lstm = np.corrcoef(omega_pc_list,del_lstm)[0,1]

    fig,ax = plt.subplots(1)

    ax.plot(omega_pc_list,del_of,'.-',markersize = markersize,linewidth = linewidth,label = 'OpenFAST')
    ax.plot(omega_pc_list,del_n4sid,'.-',markersize = markersize,linewidth = linewidth,label = 'n4sid')
    ax.plot(omega_pc_list,del_lstm,'.-',markersize = markersize,linewidth = linewidth,label = 'LSTM')


    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel)
    ax.set_ylabel('DEL [kNm]',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 3,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        fig.savefig(plot_path +os.sep+ 'DEL_comp.pdf')
    plt.close(fig)


    plt.show()



    breakpoint()