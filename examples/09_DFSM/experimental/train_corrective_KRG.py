import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.cluster import KMeans

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,GRU
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU,Dense,TimeDistributed
from tqdm import tqdm

import sherpa
import tensorflow.keras.activations as F
import smt.surrogate_models as smt
#from sherpa.algorithms import GPyOpt


if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_folder = this_dir + os.sep + 'outputs'

    results_file = outputs_folder+os.sep+'CL_val_iea22_train_18'+os.sep +'ts_dict.pkl'

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

    save_folder = outputs_folder + os.sep +'corrective_fun_results_iea22_18'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    sf = save_folder + os.sep + 'train_KRG'
    if not os.path.exists(sf):
        os.mkdir(sf)

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
    
    print(n_cases)
    nw = 1; ns = 5
    ind_train = np.arange(0,4)
    n_train = len(ind_train)

    # remove fist intry

    wind_array = wind_array[:nt-1,:]
    wave_array = wave_array[:nt-1,:]
    myt_of = myt_of[:nt-1,:]/1e5
    myt_dfsm = myt_dfsm[:nt-1:]/1e5
    
    nt = nt-1
    
    wind_array = np.reshape(wind_array,[nt,nw,ns],order = 'C')
    wave_array = np.reshape(wave_array,[nt,nw,ns],order = 'C')
    myt_of = np.reshape(myt_of,[nt,nw,ns],order = 'C')
    myt_dfsm = np.reshape(myt_dfsm,[nt,nw,ns],order = 'C')

    wind_test = wind_array[:,:,-1]
    wave_test = wave_array[:,:,-1]
    myt_of_test = myt_of[:,:,-1]
    myt_dfsm_test = myt_dfsm[:,:,-1]


    wind_train = wind_array[:,:,ind_train]
    wave_train = wave_array[:,:,ind_train]
    myt_of_train = myt_of[:,:,ind_train]
    myt_dfsm_train = myt_dfsm[:,:,ind_train]

    myt_of_train = np.reshape(myt_of_train,[nt*nw*n_train,],order = 'F')
    myt_dfsm_train = np.reshape(myt_dfsm_train,[nt*nw*n_train,],order = 'F')
    wind_train = np.reshape(wind_train,[nt*nw*n_train,],order = 'F')
    wave_train = np.reshape(wave_train,[nt*nw*n_train,],order = 'F')

    myt_of_test = np.reshape(myt_of_test,[nt*nw,],order='F')
    myt_dfsm_test = np.reshape(myt_dfsm_test,[nt*nw,],order = 'F')
    wind_test = np.reshape(wind_test,[nt*nw,],order = 'F')
    wave_test = np.reshape(wave_test,[nt*nw,],order = 'F')



    input_train = np.vstack([wind_train,wave_train,myt_dfsm_train]).T
    input_test = np.vstack([wind_test,wave_test,myt_dfsm_test]).T

    # fig,ax = plt.subplots(1);ax.plot(input_test[:,0]);fig.savefig('wind_all.png');plt.close(fig)
    # fig,ax = plt.subplots(1);ax.plot(input_train[:,0]);fig.savefig('wind_train_all.png');plt.close(fig)

    sample_list = [100,200,300,400,500,600,700,800,1000]

    error_train = myt_of_train - input_train[:,-1]

    myt_of_train = myt_of_train.reshape(-1,1)
    error_train = error_train.reshape(-1,1)

    combined_array = np.hstack([input_train,error_train])



    for i,n_samples in enumerate(sample_list):
    

    
        # perform kmeans clustering algorithm
        kmeans = KMeans(n_clusters = n_samples,n_init = 10).fit(X = combined_array)

        # extract centroid centers and centroid index
        inputs_sampled = kmeans.cluster_centers_

        inputs_train_sampled = inputs_sampled[:,:3]
        outputs_train_sampled = inputs_sampled[:,-1]

        

        # train a surrogate model

        sm = smt.KRG(print_global=False, theta0=[1e-1])
        sm.set_training_values(inputs_train_sampled,outputs_train_sampled)
        sm.train()



        fig,ax = plt.subplots(1)
        ax.plot(input_test[:,0])
        fig.savefig(sf + os.sep +'wind_val.png')
        plt.close(fig)
        
        test_output = sm.predict_values(input_test)
        test_output = np.squeeze(test_output)
        
        test_output+= input_test[:,-1]

        time = np.arange(0,600,0.01)

        
        
        fig,ax = plt.subplots(2,1)
        ax[0].plot(time,myt_of_test,label = 'OF')
        ax[0].plot(time,test_output,label = 'Corr')
        
        ax[0].set_xlim(200,400)
        

        ax[0].legend(ncol = 2)

        ax[1].plot(time,myt_of_test,label = 'OF')
        ax[1].plot(time,input_test[:,2],alpha = 0.9,label = 'L')
        ax[1].set_xlim(200,400)

        ax[1].legend(ncol = 2)

        #plt.show()

        fig.savefig(sf + os.sep +'comp_val_'+str(i)+'.png')
        plt.close(fig)

    





    




















