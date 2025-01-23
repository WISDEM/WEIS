import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from weis.dfsm.dfsm_utilities import valid_extension,calculate_MSE
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline
from pCrunch import PowerProduction

if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    results_folder = 'validation_results_1p6'
    results_file = this_dir + os.sep +results_folder + os.sep + 'DFSM_validation_results.pkl'

    comp_fol = this_dir + os.sep +results_folder + os.sep + 'compiled_results'
    format = '.pdf'
    save_flag = True

    if not(os.path.isdir(comp_fol)):
        os.makedirs(comp_fol)

    with open(results_file,'rb') as handle:
        results_dict = pickle.load(handle)

    MSE_dict = results_dict['MSE_dict']
    mean_dict = results_dict['mean_dict']
    max_dict = results_dict['max_dict']
    min_dict = results_dict['min_dict']
    std_dict = results_dict['std_dict']
    PSD_dict = results_dict['PSD_dict']
    BldPitch_array = results_dict['BldPitch_array']
    TwrBsMyt_array = results_dict['TwrBsMyt_array']
    GenPwr_array = results_dict['GenPwr_array']
    PtfmPitch_array = results_dict['PtfmPitch_array']
    DEL_array = results_dict['DEL_array']

    n_ws = 10;n_seeds = 5;
    test_inds = np.arange(0,n_ws*n_seeds)
    n_test = len(test_inds)

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 12
    fontsize_axlabel = 16
    fontsize_tick = 10

    key_qty = ['RtVAvgxh','GenTq','BldPitch1','GenSpeed','GenPwr','TwrBsMyt','PtfmPitch']

    n_qty = len(key_qty)
    MSE_array = np.zeros((n_test,n_qty))
    MSE_tw = np.zeros(n_test)
    GP = np.zeros(n_test)
    PP = np.zeros(n_test)
    WS = np.zeros(n_test)

    DEL_of = np.zeros(n_test)
    DEL_dfsm = np.zeros(n_test)

    BP_r2 = np.zeros(n_test)
    GP_r2 = np.zeros(n_test)
    PP_r2 = np.zeros(n_test)
    TW_r2 = np.zeros(n_test)

    mean_of_array = np.zeros((n_test,n_qty))
    mean_dfsm_array = np.zeros((n_test,n_qty))

    min_of_array = np.zeros((n_test,n_qty))
    min_dfsm_array = np.zeros((n_test,n_qty))

    max_of_array = np.zeros((n_test,n_qty))
    max_dfsm_array = np.zeros((n_test,n_qty))

    std_of_array = np.zeros((n_test,n_qty))
    std_dfsm_array = np.zeros((n_test,n_qty))

    for i in test_inds:
        for iqty,qty in enumerate(key_qty):
            mse_dict_name = qty + '_'+str(i)
            dict_name_of = qty + '_of_' + str(i)
            dict_name_dfsm = qty + '_dfsm_' + str(i)

            MSE_array[i,iqty] = MSE_dict[mse_dict_name]

            mean_of_array[i,iqty] = mean_dict[dict_name_of]
            mean_dfsm_array[i,iqty] = mean_dict[dict_name_dfsm]

            min_of_array[i,iqty] = min_dict[dict_name_of]
            min_dfsm_array[i,iqty] = min_dict[dict_name_dfsm]

            max_of_array[i,iqty] = max_dict[dict_name_of]
            max_dfsm_array[i,iqty] = max_dict[dict_name_dfsm]

            std_of_array[i,iqty] = std_dict[dict_name_of]
            std_dfsm_array[i,iqty] = std_dict[dict_name_dfsm]

        dict = TwrBsMyt_array[i]
        MSE_tw[i] = calculate_MSE(dict['OpenFAST']/100,dict['DFSM']/100)
        TW_r2[i] = r2_score(dict['OpenFAST'],dict['DFSM'])

        dict_del = DEL_array[i]
        DEL_of[i] = dict_del['OpenFAST']
        DEL_dfsm[i] = dict_del['DFSM']


        dict2 = GenPwr_array[i]
        GP[i] = calculate_MSE(dict2['OpenFAST']/10,dict2['DFSM']/10)
        GP_r2[i] = r2_score(dict2['OpenFAST'],dict2['DFSM'])

        dict3 = PtfmPitch_array[i]
        PP[i] = calculate_MSE(dict3['OpenFAST']*10,dict3['DFSM']*10)
        PP_r2[i] = r2_score(dict3['OpenFAST'],dict3['DFSM'])

        dict4 = BldPitch_array[i]
        BP_r2[i] = r2_score(dict4['OpenFAST'],dict4['DFSM'])

        WS[i] = mean_dict['RtVAvgxh_of_'+str(i)]

            
    MSE_array = np.reshape(MSE_array,[n_qty,n_seeds,n_ws])
    MSE_array = np.mean(MSE_array,axis = 1)

    MSE_tw = np.reshape(MSE_tw,[n_ws,n_seeds])
    MSE_tw = np.mean(MSE_tw,axis = 1)

    GP = np.reshape(GP,[n_ws,n_seeds])
    GP = np.mean(GP,axis = 1)

    PP = np.reshape(PP,[n_ws,n_seeds])
    PP = np.mean(PP,axis = 1)

    WS = np.reshape(WS,[n_ws,n_seeds])
    WS = np.mean(WS,axis = 1)
    
    mean_of_array = np.reshape(mean_of_array,[n_ws,n_seeds,n_qty],order = 'C')
    mean_of_array = np.mean(mean_of_array,axis = 1)

    min_of_array = np.reshape(min_of_array,[n_ws,n_seeds,n_qty],order = 'C')
    min_of_array = np.mean(min_of_array,axis = 1)

    max_of_array = np.reshape(max_of_array,[n_ws,n_seeds,n_qty],order = 'C')
    max_of_array = np.mean(max_of_array,axis = 1)


    mean_dfsm_array = np.reshape(mean_dfsm_array,[n_ws,n_seeds,n_qty],order = 'C')
    mean_dfsm_array = np.mean(mean_dfsm_array,axis = 1)

    min_dfsm_array = np.reshape(min_dfsm_array,[n_ws,n_seeds,n_qty],order = 'C')
    min_dfsm_array = np.mean(min_dfsm_array,axis = 1)

    max_dfsm_array = np.reshape(max_dfsm_array,[n_ws,n_seeds,n_qty],order = 'C')
    max_dfsm_array = np.mean(max_dfsm_array,axis = 1)

    std_of_array = np.reshape(std_of_array,[n_ws,n_seeds,n_qty],order = 'C')
    std_of_array = np.mean(std_of_array,axis = 1)
    std_dfsm_array = np.reshape(std_dfsm_array,[n_ws,n_seeds,n_qty],order = 'C')
    std_dfsm_array = np.mean(std_dfsm_array,axis = 1)

    GP_r2 = np.reshape(GP_r2,[n_ws,n_seeds])
    GP_r2m = np.mean(GP_r2,axis = 1)

    PP_r2 = np.reshape(PP_r2,[n_ws,n_seeds])
    PP_r2m = np.mean(PP_r2,axis = 1)
    
    BP_r2 = np.reshape(BP_r2,[n_ws,n_seeds])
    BP_r2m = np.mean(BP_r2,axis = 1)

    TW_r2 = np.reshape(TW_r2,[n_ws,n_seeds])
    TW_r2m = np.mean(TW_r2,axis = 1)

    CS = np.round(mean_of_array[:,0],1)

    DEL_of = np.reshape(DEL_of,[n_ws,n_seeds])
    DEL_of = np.mean(DEL_of,axis = 1)

    DEL_dfsm = np.reshape(DEL_dfsm,[n_ws,n_seeds])
    DEL_dfsm = np.mean(DEL_dfsm,axis = 1)

    pavg_of = mean_of_array[:,4]
    pavg_dfsm = mean_dfsm_array[:,4]

    
    pp = PowerProduction('I')
    ws_prob = pp.prob_WindDist(CS, disttype='pdf')
    ws_prob /= ws_prob.sum()


    DEL_of_wt = DEL_of*ws_prob
    DEL_dfsm_wt = DEL_dfsm*ws_prob
    #-----------------------------------------------------------------
    # WS vs weighted DEL
    #-----------------------------------------------------------------

    fig,ax = plt.subplots(1)
    ax.plot(WS,DEL_of*ws_prob,'.-',markersize = 14,label = 'OpenFAST')
    ax.plot(WS,DEL_dfsm*ws_prob,'.-',markersize = 14,label = 'DFSM')
    ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 2,fontsize = fontsize_legend)
    ax.set_title('Weighted DEL',fontsize = fontsize_axlabel)
    fig_name = 'WSvsDEL'

    if save_flag:     
        fig.savefig(comp_fol +os.sep+ fig_name + format)

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
    ax.text(10000,25000,text,size = fontsize_legend)
    ax.set_xlabel('DEL OpenFAST',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_ylabel('DEL DFSM',fontsize = fontsize_axlabel)
    fig_name = 'DELvsDEL'

    if save_flag:     
        fig.savefig(comp_fol +os.sep+ fig_name + format)

    #-----------------------------------------------------------------
    # WS vs. weighted Pavg
    #-----------------------------------------------------------------

    pavg_of_wt = pavg_of*ws_prob
    pavg_dfsm_wt = pavg_dfsm*ws_prob

    AEP_of = round(np.trapz(pavg_of_wt,WS)*8760/1e6,1)
    AEP_dfsm = round(np.trapz(pavg_dfsm_wt,WS)*8760/1e6,1)


    fig,ax = plt.subplots(1)
    ax.plot(WS,pavg_of_wt,'.-',markersize = 14,label = 'OpenFAST')
    ax.plot(WS,pavg_dfsm_wt,'.-',markersize = 14,label = 'DFSM')
    ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(ncol = 2,fontsize = fontsize_legend)
    ax.set_ylabel('Weighted P_avg [kW]',fontsize = fontsize_axlabel)
    fig_name = 'WSvsPavg'

    if save_flag:     
        fig.savefig(comp_fol +os.sep+ fig_name + format)

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
    ax.text(500,2000,text_r2,size = fontsize_legend)
    ax.text(500,1750,text_aep_of,size = fontsize_legend)
    ax.text(500,1500,text_aep_dfsm,size = fontsize_legend)
    ax.set_xlabel('Weighted P_avg OpenFAST',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)
    #ax.legend(ncol = 2,fontsize = fontsize_legend)
    ax.set_ylabel('Weighted P_avg DFSM',fontsize = fontsize_axlabel)
    fig_name = 'PavgvsPavg'

    if save_flag:     
        fig.savefig(comp_fol +os.sep+ fig_name + format)



    #-----------------------------------------------------------------
    # WS vs. range of TwrBSMyt
    #-----------------------------------------------------------------
    fmt ='-'
    capsize = 4
    alpha = 0.9

    for ind,qty in enumerate(key_qty):

        if not(qty in ['RtVAvgxh','GenTq']): 


            qty_of_lower = mean_of_array[:,ind] - min_of_array[:,ind]
            qty_of_upper = max_of_array[:,ind] - mean_of_array[:,ind]
            qty_of_mean = mean_of_array[:,ind]

            qty_dfsm_lower = mean_dfsm_array[:,ind] - min_dfsm_array[:,ind]
            qty_dfsm_upper = max_dfsm_array[:,ind] - mean_dfsm_array[:,ind]
            qty_dfsm_mean = mean_dfsm_array[:,ind]

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
            
            ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_title(qty,fontsize = fontsize_axlabel)

            fig_name = qty+'_range'

            if save_flag:     
                fig.savefig(comp_fol +os.sep+ fig_name + format)

    plt.show()


