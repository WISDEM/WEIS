import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from weis.dfsm.dfsm_utilities import calculate_MSE

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,GRU
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU,Dense,TimeDistributed
from tqdm import tqdm

import sherpa
import tensorflow.keras.activations as F
import fatpack
from scipy.stats import weibull_min
#from sherpa.algorithms import GPyOpt

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 15

slope = 4
bins = 100

def reshape_mean(array,ns,nw):
    A = np.reshape(array,[ns,nw],order = 'F')

    A = np.mean(A,axis = 0)

    return A

def calc_DEL(myt,elapsed):

    F, Fmean = fatpack.find_rainflow_ranges(myt, return_means=True)
    Nrf, Frf = fatpack.find_range_count(F, bins)
    DELs_= Frf ** slope * Nrf / elapsed

    DEL = DELs_.sum() ** (1.0 / slope)

    return DEL




if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    results_file = this_dir + os.sep + 'ts_dict_test_15s.pkl'

    # load results file
    with open(results_file,'rb') as handle:
        results_dict = pickle.load(handle) 

    # load results
    wind_list = results_dict['ws_array']
    wave_list = results_dict['wave_array']
    myt_list = results_dict['myt_array']

    # get number of cases
    n_cases = len(wind_list)

    # get number of time points
    nt = len(wind_list[0]['RtVAvgxh'])

    RNN_type = 'LSTM'

    save_folder = this_dir + os.sep +'corrective_fun_results_test_15s_test' + RNN_type
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # initialize storage arrays
    wind_array = np.zeros((nt,n_cases))
    wave_array = np.zeros((nt,n_cases))
    myt_of = np.zeros((nt,n_cases))
    myt_dfsm = np.zeros((nt,n_cases))

    for i in range(n_cases):

        wind_array[:,i] = wind_list[i]['RtVAvgxh']
        wave_array[:,i] = wave_list[i]['Wave1Elev']
        myt_of[:,i] = myt_list[i]['OpenFAST']
        myt_dfsm[:,i] = myt_list[i]['DFSM']

    nw = 7; ns = 15

    k = 2
    v_avg = 11.4
    from scipy.special import gamma

    TI = np.zeros((n_cases))

    for i in range(n_cases):
        TI[i] = np.std(wind_array[:,i])/np.mean(wind_array[:,i])

    mean_wind = np.mean(wind_array,axis = 0);

    mw = np.reshape(mean_wind,[ns,nw],order = 'F')
    mw = np.mean(mw,axis = 0)

    scale = v_avg / gamma(1.0 + 1.0 / k)
    prob = weibull_min.pdf(mw, k, loc=0, scale=scale)
    prob = prob/np.sum(prob)


    fig,ax = plt.subplots(1);
    ax.plot(mean_wind,TI,'.',color = 'k',markersize = 10)
    ax.set_xlabel('Wind Speed [m/s]');ax.set_ylabel('TI')
    fig.savefig(save_folder+os.sep + 'TI.png')
    plt.close(fig)
    #plt.show()

    # remove fist intry

    wind_array = wind_array[:nt-1,:]
    wave_array = wave_array[:nt-1,:]
    myt_of = myt_of[:nt-1,:]/1e5
    myt_dfsm = myt_dfsm[:nt-1:]/1e5
    
    nt = nt-1
    print(nt)

    save_name = RNN_type + '_corrective_test.keras'
    
    if os.path.exists(save_name):

        model = load_model(save_name)

    time = np.arange(0,600,0.01)

    elapsed = time[-1] - time[0]

    nt_train_list = [1000,60000]

    xlim = [100,400]
    

    for i_nt,nt_train in enumerate(nt_train_list):

        batch_size = int(nt/nt_train)

        DEL = np.zeros((n_cases,3))
        M_std = np.zeros((n_cases,3))

        if nt_train == 1000:
            n_cases_ = 1

        else:
            n_cases_ = n_cases

        for i_case in range(n_cases_):
            
            test_output = np.zeros((batch_size,nt_train))

            input_ = np.vstack([wind_array[:,i_case],wave_array[:,i_case],myt_dfsm[:,i_case]]).T
            
            input_ = input_.reshape([batch_size,nt_train,3],order = 'C')

            test_output = model.predict(input_)
            test_output =np.reshape(test_output,[nt,],order = 'C')

            
            fig,ax = plt.subplots(1)

            ax.plot(time,myt_of[:,i_case],label = 'OpenFAST', color = 'k')
            ax.plot(time,test_output,label = 'DFSM + C',color = 'tab:orange')
            ax.set_xlabel(['Time [s]'],fontsize = fontsize_axlabel)
            ax.set_ylabel('Norm. TwrBsMyt',fontsize = fontsize_axlabel)
            ax.set_xlim(xlim)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)

            #plt.show()
            fig.savefig(save_folder + os.sep +'comp_corr'+str(i_case)+'.pdf')
            plt.close(fig)

            fig,ax = plt.subplots(1)

            ax.plot(time,myt_of[:,i_case],label = 'OpenFAST',color = 'k')
            ax.plot(time,myt_dfsm[:,i_case],label = 'DFSM',color = 'red')
            ax.set_xlabel(['Time [s]'],fontsize = fontsize_axlabel)
            ax.set_ylabel('Norm. TwrBsMyt',fontsize = fontsize_axlabel)
            ax.set_xlim(xlim)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)

            fig.savefig(save_folder + os.sep +'comp_LPV'+str(i_case)+'.pdf')
            plt.close(fig)

            DEL[i_case,0] = calc_DEL(myt_of[:,i_case]*1e0,elapsed)
            DEL[i_case,1] = calc_DEL(myt_dfsm[:,i_case]*1e0,elapsed)
            DEL[i_case,2] = calc_DEL(test_output*1e0,elapsed)

            M_std[i_case,0] = np.std(myt_of[:,i_case])
            M_std[i_case,1] = np.std(myt_dfsm[:,i_case])
            M_std[i_case,2] = np.std(test_output)

        DEL_of = reshape_mean(DEL[:,0],ns,nw)*prob
        DEL_dfsm = reshape_mean(DEL[:,1],ns,nw)*prob
        DEL_corr = reshape_mean(DEL[:,2],ns,nw)*prob
        #--------------------------------------------------
        fig,ax = plt.subplots(1)

        ax.plot(mw,DEL_of,'ko-',label = 'OpenFAST')
        #ax.plot(mw,DEL_dfsm,'ro-',label = 'DFSM')
        ax.plot(mw,DEL_corr,'o-',color = 'tab:orange', label = 'DFSM + Corr')

        ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
        ax.set_ylabel('Weighted DEL',fontsize = fontsize_axlabel)
        ax.tick_params(labelsize=fontsize_tick)
        ax.legend(ncol = 2,fontsize = fontsize_legend)

        fig.savefig(save_folder + os.sep +'DEL_corr.pdf')
        plt.close(fig)
        #---------------------------------------------------
        fig,ax = plt.subplots(1)

        ax.plot(mw,DEL_of,'ko-',label = 'OpenFAST')
        ax.plot(mw,DEL_dfsm,'ro-',label = 'DFSM')

        ax.tick_params(labelsize=fontsize_tick)
        ax.legend(ncol = 2,fontsize = fontsize_legend)
        ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
        ax.set_ylabel('Weighted DEL',fontsize = fontsize_axlabel)

        fig.savefig(save_folder + os.sep +'DEL_lpv.pdf')
        plt.close(fig)
        #--------------------------------------------------------------
        fig,ax = plt.subplots(1)

        DEL_of_ = np.sort(DEL_of)
        sort_ind = np.argsort(DEL_of)

        ax.plot(DEL_of_,DEL_of[sort_ind],'ko-',label = 'OpenFAST')
        ax.plot(DEL_of_,DEL_dfsm[sort_ind],'ro-',label = 'DFSM')
        ax.plot(DEL_of_,DEL_corr[sort_ind],'o-',color = 'tab:orange', label = 'DFSM + Corr')

        ax.legend(ncol = 3,fontsize = fontsize_legend)
        ax.tick_params(labelsize=fontsize_tick)
        ax.set_xlabel('Actual DEL',fontsize = fontsize_axlabel)
        ax.set_ylabel('Predicted DEL',fontsize = fontsize_axlabel)

        fig.savefig(save_folder + os.sep +'act-pred.pdf')
        plt.close(fig)
        #-------------------------------------------------
        fig,ax = plt.subplots(1)

        ax.plot(mw,reshape_mean(M_std[:,0],ns,nw),'ko-',label = 'OpenFAST')
        ax.plot(mw,reshape_mean(M_std[:,1],ns,nw),'ro-',label = 'DFSM')
        ax.plot(mw,reshape_mean(M_std[:,2],ns,nw),'o-',color = 'tab:orange', label = 'DFSM + Corr')

        ax.legend(ncol = 3)

        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_ylabel('Myt std')

        fig.savefig(save_folder + os.sep +'STD.pdf')
        plt.close(fig)

        print('---------------------------------------')

        print(np.sum(DEL_of))
        print(np.sum(DEL_dfsm))
        print(np.sum(DEL_corr))
        #model.save(save_name)





    




















