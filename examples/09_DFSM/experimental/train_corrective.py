import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,GRU
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU,Dense,TimeDistributed
from tqdm import tqdm

import sherpa
import tensorflow.keras.activations as F
#from sherpa.algorithms import GPyOpt

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_folder = this_dir + os.sep + 'outputs'

    results_file = outputs_folder+os.sep+'RM1_train'+os.sep +'ts_dict.pkl'

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

    save_folder = outputs_folder + os.sep +'corrective_fun_results_RM1'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    sf = save_folder + os.sep + 'train'
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
    nw = 10; ns = 6
    ind_train = np.arange(0,5)
    n_train = len(ind_train)

    # remove fist intry

    wind_array = wind_array[:nt-1,:]
    wave_array = wave_array[:nt-1,:]
    myt_of = myt_of[:nt-1,:]/1e3
    myt_dfsm = myt_dfsm[:nt-1:]/1e3
    
    
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

    el_per_batch = 500
    

    batch_size_train = int(len(myt_of_train)/el_per_batch);
    batch_size_test = int(len(myt_of_test)/el_per_batch);
    

    myt_of_train = np.reshape(myt_of_train,[batch_size_train,el_per_batch,1],order = 'C');
    myt_dfsm_train = np.reshape(myt_dfsm_train,[batch_size_train,el_per_batch,1],order = 'C');
    input_train = np.reshape(input_train,[batch_size_train,el_per_batch,3],order = 'C');
    error_train = myt_of_train - myt_dfsm_train

    myt_of_test = np.reshape(myt_of_test,[batch_size_test,el_per_batch,1],order = 'C');
    myt_dfsm_test = np.reshape(myt_dfsm_test,[batch_size_test,el_per_batch,1],order = 'C');
    input_test = np.reshape(input_test,[batch_size_test,el_per_batch,3],order = 'C');
    error_test = myt_of_test - myt_dfsm_test
    


    
    model = tf.keras.Sequential()
    n_cells = 64
    n_epoch = 5

    RNN_type = 'LSTM'
    save_name = save_folder + os.sep +RNN_type + '_corrective.keras'
    
    if os.path.exists(save_name):

        model = load_model(save_name)

    else:

        if RNN_type == 'LSTM':

            model.add(LSTM(n_cells,return_sequences = True,input_shape = (None,3)))

        elif RNN_type == 'GRU':

            model.add(GRU(n_cells,return_sequences = True,input_shape = (None,3)))

        elif RNN_type == 'LSTM3':

            model.add(LSTM(n_cells,return_sequences = True,input_shape = (None,3)))
            model.add(LSTM(n_cells,return_sequences = True,input_shape = (None,3)))

        model.add(TimeDistributed(Dense(1)))

        model.compile(optimizer = 'adam',loss = 'mse')
        model.fit(input_train,myt_of_train,batch_size = 1,epochs = n_epoch,verbose = 2)

        model.save(save_name)

    

    for i_batch in range(batch_size_test):

        input_ = input_test[i_batch,:,:]
        input_ = input_.reshape([1,el_per_batch,3],order = 'F')

        fig,ax = plt.subplots(1)
        ax.plot(input_[0,:,0])
        fig.savefig(sf + os.sep +'wind_test'+str(i_batch)+'.png')
        plt.close(fig)
        
        test_output = model.predict(input_)

        test_output = np.squeeze(test_output)

        
        
        fig,ax = plt.subplots(2,1)

        ax[0].plot(myt_of_test[i_batch,:],label = 'OF')
        ax[0].plot(input_[0,:,2],label = 'L')

        ax[0].legend(ncol = 2)

        ax[1].plot(myt_of_test[i_batch,:],label = 'OF')
        ax[1].plot(test_output,alpha = 0.9,label = 'L + corr')

        ax[1].legend(ncol = 2)


        #plt.show()

        fig.savefig(sf + os.sep +'comp_'+str(i_batch)+'.png')
        plt.close(fig)

    model.save(save_name)





    




















