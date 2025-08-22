import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle

from scipy.interpolate import CubicSpline,interp1d


import time as timer

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension
from weis.dfsm.dfsm_sample_data import sample_data

import matlab.engine

def ModelData():
    
    model_data = {}

    model_data['reqd_states'] = ['PtfmPitch','TTDspFA','GenSpeed'];ns = len(model_data['reqd_states'])
    model_data['reqd_controls'] =  ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev'];nc = len(model_data['reqd_controls'])
    model_data['reqd_outputs'] = ['TwrBsMyt','NcIMURAys','GenSpeed'] 

    model_data['datapath'] = '/home/athulsun/DFSM/data' + os.sep + 'FOWT_1p6'

    scale_args = {'state_scaling_factor': np.array([1,1,1]),
                  'control_scaling_factor': np.array([1,1000,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1,1])
                  }
    
    model_data['scale_args'] = scale_args

    model_data['nx'] = 2

    # indices to be used for training and testing
    model_data['test_inds'] = np.arange(5,10)
    
    
    # current speed to start the optimization
    model_data['w_test'] = 18
    
    # training indices
    model_data['train_inds'] = np.arange(0,5)

    # pickle file to save the DFSM to
    model_data['dfsm_file_name'] = 'fowt_1p6_n4sid.pkl'

    # Use simulations from tmin seconds
    model_data['tmin'] = 00

    model_data['dt_extract'] = 0.1
     
    return model_data

def call_matlab(inputs,outputs,nc,ny,outputs_max,n_tests,nx,nt,dt):

    print('started matlab engine')

    # Start the MATLAB Engine
    eng = matlab.engine.start_matlab()

        
    inputs_ = matlab.double(inputs)
    outputs_ = matlab.double(outputs)
    outputs_max_ = matlab.double(outputs_max)

    nx_ = matlab.double(nx)
    nc_ = matlab.double(nc)
    ny_ = matlab.double(ny)
    dt_ = matlab.double(dt)
    nt_ = matlab.double(nt)

    n_test = matlab.double(n_tests)


    t1 = timer.time()        
    A,B,C,x0 = eng.construct_LTI_n4sid(inputs_,outputs_,nc_,ny_,outputs_max_,n_tests,nx_,nt_,dt_,nargout = 4)
    t2 = timer.time()

    # stop timer
    model_construct_time = t2-t1


    print('shutting matlab down')
    eng.quit()

    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    x0 = np.array(x0)

    return A,B,C,x0
    


if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # datapath
    region = 'LPV'

    model_data = ModelData()
    datapath =  model_data['datapath'] #this_dir + os.sep + 'outputs' + os.sep + 'FOWT_1p6' #+ os.sep + 'openfast_runs/rank_0'
    
    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)

    # required states
    reqd_states = model_data['reqd_states'] 
    reqd_controls = model_data['reqd_controls']
    reqd_outputs = model_data['reqd_outputs']

    ns = len(reqd_states)
    nc = len(reqd_controls)
    ny = len(reqd_outputs)

    nx = model_data['nx']
    dt_extract = model_data['dt_extract']
    
    
    # scaling parameters
    scale_args = model_data['scale_args']
    
    # filter parameters
    filter_args = {'state_filter_flag': [False]*ns,
                   'state_filter_type': [['filtfilt']]*ns,
                   'state_filter_tf': [[0.1]]*ns
                   }

    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = None,region = region,dt_extract = dt_extract)
    
    # load and process data
    sim_detail.load_openfast_sim()
    
    
    # extract data
    FAST_sim = sim_detail.FAST_sim

    FAST_sim_array = np.array(FAST_sim)


    w_array = []

    for sim_det in FAST_sim_array:
        w_array.append(sim_det['w_mean'])

    w_array = np.round(np.array(w_array))
    W = np.unique(w_array)
    
    nW = len(W)
    nseeds = int(len(w_array)/nW)
    
    FAST_sim = np.reshape(FAST_sim_array,[nseeds,nW],order = 'F')
    w_array = np.reshape(w_array,[nseeds,nW],order = 'F')

    sort_ind = np.argsort(w_array[0,:])
    FAST_sim = FAST_sim[:,sort_ind]

    n_samples = 1

    w_test = model_data['w_test']

    ind_w = np.where(W == w_test)[0][0]

    train_inds = model_data['train_inds']
    test_inds = model_data['test_inds']
    

    _,_,_,inputs,state_dx,outputs = sample_data(FAST_sim[train_inds,ind_w],'KM',n_samples = 1)
    inputs = np.array(inputs[:,:nc])
    
    outputs_max = np.abs(np.max(outputs,0))
    max_ind = outputs_max < 1e-4
    outputs_max[max_ind] = 1 

    nt = len(FAST_sim[0,0]['controls'])

    n_tests = len(train_inds)

    A,B,C,x0 = call_matlab(inputs,outputs,nc,ny,outputs_max,n_tests,nx,nt,dt_extract)

    # load results from matlab
    save_flag = True;save_dict_flag = True;plot_flag = True
    plot_path = 'sys_id_test'

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 18
    fontsize_tick = 12

    
    for idx,ind in enumerate(test_inds):
        

        FS = FAST_sim[ind,ind_w]

        
        U_of = FS['controls']
        outputs_OF = FS['outputs']

        #time = results_mat['time']
        time = FAST_sim[ind,ind_w]['time']
        t_ind = time>=100

        nt = len(time)

        # initalize
        X = np.zeros((nt+1,nx))
        Y = np.zeros((nt,ny))
        
        X[0,:] = x0[:,1]

        for i in range(nt):

            u = U_of[i,:]
            x = X[i,:]

            xi = np.dot(A,x) + np.dot(B,u)

            Y[i,:] = np.dot(C,x)
            X[i+1,:] = xi
        breakpoint()
        Y = Y*outputs_max
        if plot_flag:


            fig,ax = plt.subplots(1)
            ax.plot(time[t_ind],U_of[t_ind,0])
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            ax.set_title('RtVAvgxh',fontsize = fontsize_axlabel)

            if save_flag:
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
        
                fig.savefig(plot_path +os.sep+ 'RtVAvgxh_' +str(idx)+ '.pdf')
            plt.close(fig)



            for i in range(ny):

                fig,ax1 = plt.subplots(1)
                
                ax1.plot(time[t_ind],outputs_OF[t_ind,i],label = 'OpenFAST')
                ax1.plot(time[t_ind],Y[t_ind,i],label = 'n4sid',alpha = 0.8)
                ax1.set_xlim([100,700])
                
                ax1.tick_params(labelsize=fontsize_tick)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                ax1.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                ax1.set_title(reqd_outputs[i],fontsize = fontsize_axlabel)

                if save_flag:
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
            
                    fig.savefig(plot_path +os.sep+ reqd_outputs[i] +'_' +str(idx)+ '_comp.pdf')
                plt.close(fig)


    plt.show()

    dfsm_python_pkl = model_data['dfsm_file_name']

    with open(dfsm_python_pkl,'wb') as handle:
        pickle.dump({'A':A,'B':B,'C':C,'x0':x0},handle)

    




        