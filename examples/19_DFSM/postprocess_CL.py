import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from weis.dfsm.dfsm_utilities import valid_extension,calculate_MSE
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline

speed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
probability = [0.176358711313469, 0.225347242233877, 0.233415941444297, 0.231686934470636, 0.210362515128811, 0.213820529076134, 0.240331969338943, 0.228228920523313, 0.245518990259927, 0.245518990259927, 0.253011353812460, 0.275488444470059, 0.293354849864562, 0.327934989337790, 0.352717422626938, 0.374041841968762, 0.391908247363264, 0.447812806178319, 0.501412022361823, 0.577488329202929, 0.566537951703071, 0.586709699729122, 0.515244078151115, 0.510633392888018, 0.442625785257335, 0.383839548152844, 0.301423549074981, 0.262809059996542, 0.169442683418823, 0.0829923347357504, 0.0547518874992795, 0.0443778456573108, 0.0190190767102761, 0.00806869921042014, 0.00403434960521009, 0.00172900697366146]
format = '.pdf'
save_flag = False

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 12

def plot_range(sum_stats_of,sum_stats_dfsm,key_qtys,CS,range_path):

    '''
    Function to loop through and plot the mean value and range of the signals in the key_qtys list
    '''

    fmt ='-'
    capsize = 4
    alpha = 0.9

    for qty in key_qtys:

        # get mean, min and max, reshape and find the average
        mean_of_array = sum_stats_of[qty]['mean'];mean_of_array = np.reshape(mean_of_array,[n_seeds,n_cs],order = 'F');mean_of_array = np.mean(mean_of_array,axis = 0)
        min_of_array = sum_stats_of[qty]['min'];min_of_array = np.reshape(min_of_array,[n_seeds,n_cs],order = 'F');min_of_array = np.mean(min_of_array,axis = 0)
        max_of_array = sum_stats_of[qty]['max'];max_of_array = np.reshape(max_of_array,[n_seeds,n_cs],order = 'F');max_of_array = np.mean(max_of_array,axis = 0)

        mean_dfsm_array = sum_stats_dfsm[qty]['mean'];mean_dfsm_array = np.reshape(mean_dfsm_array,[n_seeds,n_cs],order = 'F');mean_dfsm_array = np.mean(mean_dfsm_array,axis = 0)
        min_dfsm_array = sum_stats_dfsm[qty]['min'];min_dfsm_array = np.reshape(min_dfsm_array,[n_seeds,n_cs],order = 'F');min_dfsm_array = np.mean(min_dfsm_array,axis = 0)
        max_dfsm_array = sum_stats_dfsm[qty]['max'];max_dfsm_array = np.reshape(max_dfsm_array,[n_seeds,n_cs],order = 'F');max_dfsm_array = np.mean(max_dfsm_array,axis = 0)

        # find lower and upper error limits
        qty_of_lower = mean_of_array - min_of_array
        qty_of_upper = max_of_array - mean_of_array
        qty_of_mean = mean_of_array

        qty_dfsm_lower = mean_dfsm_array - min_dfsm_array
        qty_dfsm_upper = max_dfsm_array - mean_dfsm_array
        qty_dfsm_mean = mean_dfsm_array
        
        # plot
        fig,ax = plt.subplots(1)
        ax.errorbar(CS,qty_of_mean,
                                    yerr = [qty_of_lower,qty_of_upper],
                                    fmt =fmt, capsize=capsize, alpha = alpha)
            
        ax.errorbar(CS,qty_dfsm_mean,
                                    yerr = [qty_dfsm_lower,qty_dfsm_upper],
                                    fmt =fmt, capsize=capsize, alpha = alpha)
            
        ax.scatter(CS,qty_of_mean,marker = 'o',
                            s = 20, color = 'tab:blue')
            
        ax.scatter(CS,qty_dfsm_mean,marker = 'o',
                            s = 20, color = 'tab:orange')
        
        ax.set_xlabel('Current Speed [m/s]',fontsize = fontsize_axlabel)
        ax.tick_params(labelsize=fontsize_tick)
        ax.set_title(qty,fontsize = fontsize_axlabel)

        fig_name = qty+'_range'

        fig.savefig(range_path + os.sep + fig_name + format)
        plt.close(fig)

if __name__ == '__main__':

    # path to this directory
    this_dir= os.path.dirname(os.path.abspath(__file__))
    save_flag = True

    # path to the directory with the closed-loop validation results
    results_folder = this_dir +os.sep+ 'outputs'+ os.sep +'closed_loop_validation'

    # stored results
    results_file = results_folder + os.sep + 'DFSM_validation_results.pkl'

    # load results
    with open(results_file,'rb') as handle:
        results_dict = pickle.load(handle)

    # folder to plot the range, DEL and AEP
    range_path = results_folder + os.sep +'range'

    if not os.path.isdir(range_path):
        os.makedirs(range_path)

    # key quantities
    key_qtys = ['BldPitch1','GenSpeed','GenPwr','TwrBsMyt','PtfmPitch']
   
    sum_stats_of = results_dict['sum_stats_of']
    sum_stats_dfsm = results_dict['sum_stats_dfsm']


    CS = sum_stats_of['RtVAvgxh']['mean']

    DELs_of = results_dict['DELs_of']
    DELs_dfsm = results_dict['DELs_dfsm']

    pavg_of = sum_stats_of['GenPwr']['mean']
    pavg_dfsm = sum_stats_dfsm['GenPwr']['mean']

    n_cs = 7
    n_seeds = 6

    CS = [1.5,1.75,2.0,2.25,2.5,2.75,3.0]

    
    # load DELs and reshape
    DEL_of = np.reshape(DELs_of,[n_seeds,n_cs],order = 'F')
    DEL_of = np.mean(DEL_of,axis = 0)

    DEL_dfsm = np.reshape(DELs_dfsm,[n_seeds,n_cs],order = 'F')
    DEL_dfsm = np.mean(DEL_dfsm,axis = 0)

    pavg_of = np.reshape(pavg_of,[n_seeds,n_cs],order = 'F')
    pavg_of = np.mean(pavg_of,axis = 0)

    pavg_dfsm = np.reshape(pavg_dfsm,[n_seeds,n_cs],order = 'F')
    pavg_dfsm = np.mean(pavg_dfsm,axis = 0)

    # current speed probability
    ws_prob_fun = CubicSpline(speed,probability)

    ws_prob = ws_prob_fun(CS) 
    ws_prob /= ws_prob.sum()

    DEL_of_wt = DEL_of*ws_prob
    DEL_dfsm_wt = DEL_dfsm*ws_prob



    #-----------------------------------------------------------------
    # WS vs weighted DEL
    #-----------------------------------------------------------------

    fig,ax = plt.subplots(1)
    ax.plot(CS,DEL_of*ws_prob,'.-',markersize = 14,label = 'OpenFAST')
    ax.plot(CS,DEL_dfsm*ws_prob,'.-',markersize = 14,label = 'DFSM')
    ax.set_xlabel('Current Speed [m/s]',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 2,fontsize = fontsize_legend)
    ax.set_title('Weighted DEL',fontsize = fontsize_axlabel)
    fig_name = 'WSvsDEL'

    if save_flag:     
        fig.savefig(range_path +os.sep+ fig_name + format)

    #-----------------------------------------------------------------
    # weighted DEL_of vs DEL_dfsm
    #-----------------------------------------------------------------

    fig,ax = plt.subplots(1)

    # sort DEL
    sort_ind = np.argsort(DEL_of_wt)
    DEL_of_wt = DEL_of_wt[sort_ind]
    DEL_dfsm_wt = DEL_dfsm_wt[sort_ind]

    # calculate R2 value
    r2_del = round(r2_score(DEL_of_wt,DEL_dfsm_wt),4)
    text = 'R^2 value: ' + str(r2_del)

    # plot
    ax.plot(DEL_of_wt,DEL_dfsm_wt,'r.-',markersize = 14)
    ax.plot(DEL_of_wt,DEL_of_wt,'k--',markersize = 14,alpha = 0.7)
    ax.text(60,200,text,size = fontsize_legend)
    ax.set_xlabel('DEL OpenFAST',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_ylabel('DEL DFSM',fontsize = fontsize_axlabel)
    fig_name = 'DELvsDEL'

    if save_flag:     
        fig.savefig(range_path +os.sep+ fig_name + format)

    #-----------------------------------------------------------------
    # WS vs. weighted Pavg
    #-----------------------------------------------------------------

    pavg_of_wt = pavg_of*ws_prob
    pavg_dfsm_wt = pavg_dfsm*ws_prob

    AEP_of = round(np.trapz(pavg_of_wt,CS)*8760,1)
    AEP_dfsm = round(np.trapz(pavg_dfsm_wt,CS)*8760,1)


    fig,ax = plt.subplots(1)
    ax.plot(CS,pavg_of_wt,'.-',markersize = 14,label = 'OpenFAST')
    ax.plot(CS,pavg_dfsm_wt,'.-',markersize = 14,label = 'DFSM')
    ax.set_xlabel('Current Speed [m/s]',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 2,fontsize = fontsize_legend)
    ax.set_ylabel('Weighted P_avg [kW]',fontsize = fontsize_axlabel)
    fig_name = 'WSvsPavg'

    if save_flag:     
        fig.savefig(range_path +os.sep+ fig_name + format)

    #-----------------------------------------------------------------
    # weighted Pavg_of vs Pavg_dfsm
    #-----------------------------------------------------------------

    # sort Pavg
    sort_ind = np.argsort(pavg_of_wt)
    pavg_of_wt = pavg_of_wt[sort_ind]
    pavg_dfsm_wt = pavg_dfsm_wt[sort_ind]

    # calculate R2 value
    r2_p = round(r2_score(pavg_of_wt,pavg_dfsm_wt),4)
    text_r2 = 'R^2 value: ' + str(r2_p)
    text_aep_of = 'AEP OpenFAST: '+str(AEP_of)
    text_aep_dfsm = 'AEP DFSM: ' + str(AEP_dfsm)

    # plot
    fig,ax = plt.subplots(1)
    ax.plot(pavg_of_wt,pavg_dfsm_wt,'r.-',markersize = 14)
    ax.plot(pavg_of_wt,pavg_of_wt,'k--',markersize = 14,alpha = 0.7)
    ax.text(20,100,text_r2,size = fontsize_legend)
    ax.text(20,90,text_aep_of,size = fontsize_legend)
    ax.text(20,80,text_aep_dfsm,size = fontsize_legend)
    ax.set_xlabel('Weighted P_avg OpenFAST',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    #ax.legend(ncol = 2,fontsize = fontsize_legend)
    ax.set_ylabel('Weighted P_avg DFSM',fontsize = fontsize_axlabel)
    fig_name = 'PavgvsPavg'

    if save_flag:     
        fig.savefig(range_path +os.sep+ fig_name + format)

    plot_range(sum_stats_of,sum_stats_dfsm,key_qtys,CS,range_path)


    plt.show()




