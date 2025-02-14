import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle


import time as timer

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension
from weis.dfsm.test_dfsm import test_dfsm
from weis.dfsm.construct_dfsm import DFSM
from weis.dfsm.dfsm_sample_data import sample_data
from weis.dfsm.wrapper_LTI import wrapper_LTI

from numpy.linalg import lstsq

from weis.dfsm.dfsm_rosco_simulation import run_sim_ROSCO

import matlab.engine

def ModelData():
    
    model_data = {}

    datapath = '/home/athulsun/DFSM/data'

    # path to openfast simulation
    model_data['datapath'] = datapath + os.sep + 'RM1_train'

    # states as part of the model
    model_data['reqd_states'] = ['PtfmPitch','PtfmHeave','GenSpeed'];ns = len(model_data['reqd_states'])

    # controls/inputs
    model_data['reqd_controls'] =  ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev'];nc = len(model_data['reqd_controls'])

    # outputs used as part of the model
    model_data['reqd_outputs'] = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt'] 


    # scaling arguments
    scale_args = {'state_scaling_factor': np.array([1,1,100]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1,1,1])
                  }
    
    model_data['scale_args'] = scale_args

    # calculate number of model parameters that need to be evaluated
    n_var = ns*(nc+2*ns)

    # number of model parameters to be estimated
    model_data['n_var'] = n_var

    # indices to be used for training and testing
    model_data['test_inds'] = np.arange(5,59+6,6)
    
    
    # current speed to start the optimization
    model_data['w_start'] = 2.4
    model_data['buff'] = 0.0
    model_data['train_inds'] = np.arange(0,5)

    # pickle file to save the DFSM to
    model_data['dfsm_file_name'] = 'dfsm_mhk_all.pkl'

    # Use simulations from tmin seconds
    model_data['tmin'] = 100

     
    return model_data


if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # datapath
    region = 'LPV'

    model_data = ModelData()
    datapath =  model_data['datapath'] 
    
    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)

    # required states, controls and outputs
    reqd_states = model_data['reqd_states']
    reqd_controls = model_data['reqd_controls']
    reqd_outputs = model_data['reqd_outputs']
    
    ns = len(reqd_states)
    nc = len(reqd_controls)
    ny = len(reqd_outputs)
    
    
    # scaling parameters
    scale_args = model_data['scale_args']
    
    # filter parameters
    filter_args = {'state_filter_flag': [True]*ns,
                   'state_filter_type': [['filtfilt']]*ns,
                   'state_filter_tf': [[0.1]]*ns,
                   }
    
    # instantiate class
    tmin = model_data['tmin']
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=tmin
                                   ,add_dx2 = True,linear_model_file = None,region = region)
    
    # load and process data
    sim_detail.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail.FAST_sim

    

    n_var = model_data['n_var']
   

    FAST_sim_array = np.array(FAST_sim)

    state_names = FAST_sim_array[0]['state_names']

    # get index of nacelle rotational and translational accelerations
    ncimu_ind = reqd_outputs.index('NcIMURAys')
    yawbr_ind = reqd_outputs.index('YawBrTAxp')

    dptfmpitch_ind = state_names.index('dPtfmPitch')
    


    w_array = []

    for sim_det in FAST_sim_array:
        w_array.append(sim_det['w_mean'])

    w_array = np.round(np.array(w_array),1)
    W = np.unique(w_array)
    
    nW = len(W)
    nseeds = int(len(w_array)/nW)
    
    FAST_sim = np.reshape(FAST_sim_array,[nseeds,nW],order = 'F')
    w_array = np.reshape(w_array,[nseeds,nW],order = 'F')

    sort_ind = np.argsort(w_array[0,:])
    FAST_sim = FAST_sim[:,sort_ind]


    w_start = model_data['w_start']

    ind_start = np.where(W == w_start)[0][0]
    ind_forward_pass = np.where(W >= w_start)[0]
    ind_backward_pass = np.flip(np.where(W < w_start))[0]

    # initialize storage array
    A_array = np.zeros((nW,ns*2,ns*2))
    B_array = np.zeros((nW,2*ns,nc))
    C_array = np.zeros((nW,ny,2*ns))
    D_array = np.zeros((nW,ny,nc))

    X_array = np.zeros((nW,n_var))

    model_construct_time = np.zeros((nW,))

    train_inds = model_data['train_inds']
    test_inds = model_data['test_inds']

    # Start the MATLAB Engine
    eng = matlab.engine.start_matlab()

    X_cd_array = np.zeros((nW),dtype = object)

    LTI = wrapper_LTI(nstates = int(2*ns),ncontrols = int(nc))

    print('started matlab engine')
    
    # perform forward pass
    for ind in range(len(ind_forward_pass)):

        iw = ind_forward_pass[ind]

        _,_,_,inputs,state_dx,outputs = sample_data(FAST_sim[train_inds,iw],'KM',n_samples = 1)

        # calculate the C and D matrices
        CD = lstsq(inputs,outputs,rcond = None)
        CD = CD[0]

        CD = CD.T

        C_ = CD[:,nc:]
        D_ = CD[:,:nc]


        # convert arryas to matlab double
        inputs_md = matlab.double(inputs)
        state_dx_md = matlab.double(state_dx)
        outputs_md = matlab.double(outputs)

        n_var = matlab.double(n_var)
        ns2 = matlab.double(2*ns)
        nc_ = matlab.double(nc)
        ny_ = matlab.double(ny)

        
        if W[iw] == w_start:

            x0 = matlab.double([])
            
            t1 = timer.time()
            A,B,C,D,x = eng.construct_DFSM_matlab(x0,n_var,ns2,nc_,ny_,inputs_md,state_dx_md,outputs_md,model_data['buff'],nargout = 5)
            t2 = timer.time()
                
        else:

            x0 = matlab.double(X_array[iw-1])

            t1 = timer.time()
            A,B,C,D,x = eng.construct_DFSM_matlab(x0,n_var,ns2,nc_,ny_,inputs_md,state_dx_md,outputs_md,model_data['buff'],nargout = 5)
            t2 = timer.time()

            
        # stop timer
        model_construct_time[iw] = t2-t1

        # store model parameters
        X_array[iw,:] = np.array(x)

        # convert the A,B,C,D matrices to np arrays
        A = np.array(A);B = np.array(B)
        C = np.array(C); D = np.array(D)
        
        #if not(ptfmpitch_ind == None):
        C[ncimu_ind,:] = A[dptfmpitch_ind,:]
        D[ncimu_ind,:] = B[dptfmpitch_ind,:]

        # store linear models
        A_array[iw,:,:] = A
        B_array[iw,:,:] = B
        C_array[iw,:,:] = C
        D_array[iw,:,:] = D

    # perform backward pass
    for ind in range(len(ind_backward_pass)):

        iw = ind_backward_pass[ind]

        _,_,_,inputs,state_dx,outputs = sample_data(FAST_sim[train_inds,iw],'KM',n_samples = 1)

        inputs_md = matlab.double(inputs)
        state_dx_md = matlab.double(state_dx)
        outputs_md = matlab.double(outputs)


        # calculate the C and D matrices
        CD = lstsq(inputs,outputs,rcond = None)
        CD = CD[0]

        CD = CD.T

        C_ = CD[:,nc:]
        D_ = CD[:,:nc]

        C_array[iw,:,:] = C
        D_array[iw,:,:] = D

        x0 = matlab.double(X_array[iw+1])
        
        t1 = timer.time()
        A,B,C,D,x = eng.construct_DFSM_matlab(x0,n_var,ns2,nc_,ny_,inputs_md,state_dx_md,outputs_md,model_data['buff'],nargout = 5)
        t2 = timer.time()

        model_construct_time[iw] = t2-t1

        # store model parameters
        X_array[iw,:] = np.array(x)


        A = np.array(A);B = np.array(B)
        C = np.array(C); D = np.array(D);

        
        C[ncimu_ind,:] = A[dptfmpitch_ind,:]
        D[ncimu_ind,:] = B[dptfmpitch_ind,:]

    
        # store linear models
        A_array[iw,:,:] = A
        B_array[iw,:,:] = B
        C_array[iw,:,:] = C
        D_array[iw,:,:] = D

    print('shutting matlab down')
    eng.quit()

    interp_type = 'nearest'

    # construct surrogate model
    dfsm = DFSM(sim_detail,n_samples = 1,L_type = 'LPV',N_type = None, train_split = 0.5)

    dfsm.A_array = A_array
    dfsm.B_array = B_array
    dfsm.C_array = C_array
    dfsm.D_array = D_array
    dfsm.W = W

    dfsm.setup_LPV(interp_type)

    dfsm.AB = []
    dfsm.CD = []

    dfsm.error_ind_deriv = []
    dfsm.error_ind_outputs = []

    dfsm.dx_error = []
    dfsm.test_data = []
    dfsm.train_data =[]

    dfsm.nonlin_deriv = []
    dfsm.nonlin_outputs = []
    dfsm.nonlin_output = False
    
    
    dfsm.scaler_dx = None
    dfsm.scaler_outputs = None
    dfsm.nonlin_deriv = np.array([None])
    dfsm.nonlin_outputs = np.array([None])



    # load results from matlab
    save_flag = True;save_dict_flag = True;plot_flag = False
    plot_path = 'outputs/open_loop_validation'

    
    # test dfsm
    # flags related to testing
    simulation_flag = True 
    outputs_flag = (len(reqd_outputs) > 0)
    plot_flag = True
    
    dfsm,U_list,X_list_python,dx_list,Y_list_python = test_dfsm(dfsm,FAST_sim_array,test_inds,simulation_flag,plot_flag)
    dfsm.train_data = [];dfsm.test_data = []
    print(dfsm.simulation_time)

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 18
    fontsize_tick = 12
    
    
    for idx,ind in enumerate(test_inds):

        #time = results_mat['time']
        time = X_list_python[idx]['time']

        U = U_list[idx]['OpenFAST']

        #x_matlab = X_list_matlab[idx]['DFSM']
        x_python = X_list_python[idx]['DFSM']
        x_OF = X_list_python[idx]['OpenFAST']

        #y_matlab = Y_list_matlab[idx]['DFSM']
        y_python = Y_list_python[idx]['DFSM']
        y_OF = Y_list_python[idx]['OpenFAST']

        
        ns = len(reqd_states)
        ny = len(reqd_outputs)
        current_speed = []

        if plot_flag:

            current_speed.append(np.round(np.mean(U[:,0]),2))

            fig,ax = plt.subplots(1)
            ax.plot(time,U[:,0])
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            ax.set_title('RtVAvgxh',fontsize = fontsize_axlabel)

            if save_flag:
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
        
                fig.savefig(plot_path +os.sep+ 'RtVAvgxh_' +str(idx)+ '.pdf')
            plt.close(fig)



            for i in range(ns):

                fig,ax1 = plt.subplots(1)
                
                ax1.plot(time,x_OF[:,i],label = 'OpenFAST')
                ax1.plot(time,x_python[:,i],label = 'python',alpha = 0.8)
                
                ax1.tick_params(labelsize=fontsize_tick)
                ax1.legend(ncol = 3,fontsize = fontsize_legend)
                ax1.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                ax1.set_title(reqd_states[i],fontsize = fontsize_axlabel)

                if save_flag:
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
            
                    fig.savefig(plot_path +os.sep+ reqd_states[i] +'_' +str(idx)+ '_comp.pdf')
                plt.close(fig)

            for i in range(ny):

                fig,ax1 = plt.subplots(1)
                
                ax1.plot(time,y_OF[:,i],label = 'OpenFAST')
                ax1.plot(time,y_python[:,i],label = 'python',alpha = 0.8)
                
                ax1.tick_params(labelsize=fontsize_tick)
                ax1.legend(ncol = 3,fontsize = fontsize_legend)
                ax1.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                ax1.set_title(reqd_outputs[i],fontsize = fontsize_axlabel)

                if save_flag:
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
            
                    fig.savefig(plot_path +os.sep+ reqd_outputs[i] +'_' +str(idx)+ '_comp.pdf')
                plt.close(fig)


    dfsm_pkl = model_data['dfsm_file_name']

    with open(dfsm_pkl,'wb') as handle:
        pickle.dump(dfsm,handle)






        