import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle

from scipy.interpolate import CubicSpline,interp1d


import time as timer

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension
from weis.dfsm.test_dfsm import test_dfsm
from weis.dfsm.construct_dfsm import DFSM
from weis.dfsm.evaluate_dfsm import evaluate_dfsm
from weis.dfsm.dfsm_sample_data import sample_data
from weis.dfsm.wrapper_LTI import wrapper_LTI


from mat4py import loadmat
from numpy.linalg import lstsq

from weis.dfsm.dfsm_rosco_simulation import run_sim_ROSCO
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import argparse


import matlab.engine

def ModelData():
    
    model_data = {}

    model_data['reqd_states'] = ['PtfmPitch','TTDspFA','GenSpeed'];ns = len(model_data['reqd_states'])
    model_data['reqd_controls'] =  ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev'];nc = len(model_data['reqd_controls'])
    model_data['reqd_outputs'] = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt'] 

    model_data['datapath'] = '/home/athulsun/DFSM/data' + os.sep + 'FOWT_1p6'

    scale_args = {'state_scaling_factor': np.array([1,1,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1,1,1])
                  }
    
    model_data['scale_args'] = scale_args

    model_data['mat_file_name'] = 'FOWT_LPV_fmincon.mat'

    model_data['W'] =  np.array([3,5,7,9,11,13,15,17,19,21,23,25])

    n_var = ns*(nc+2*ns)

    model_data['n_var'] = n_var
    model_data['lb'] = [-np.inf]*n_var
    model_data['ub'] = [np.inf]*n_var


    model_data['gaopt'] = {'UseParallel':1,'Display':'iter','MaxGenerations':n_var*2}
    model_data['fminconopt'] = {'Display':'iter','Algorithm':'interior-point','MaxIterations':500,'MaxFunctionEvaluations':20000,
                                'UseParallel':1,"EnableFeasibilityMode":1,'StepTolerance':1e-8,'ConstraintTolerance':1e-4}

    model_data['nseeds'] = 10

    model_data['test_inds'] = np.array([16,26,36,46,56,66,76,86,96,106,116])

    model_data['w_start'] = 13.00
     
    return model_data


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
    
    
    # scaling parameters
    scale_args = model_data['scale_args']
    
    # filter parameters
    filter_args = {'state_filter_flag': [True]*ns,
                   'state_filter_type': [['filtfilt']]*ns,
                   'state_filter_tf': [[0.1]]*ns,
                   'control_filter_flag': [False]*nc,
                   'control_filter_tf': [0]*nc,
                   'output_filter_flag': []
                   }
    
    # name of mat file that has the linearized models
    mat_file_name = this_dir + os.sep + model_data['mat_file_name']
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    # load and process data
    sim_detail.load_openfast_sim()
    
    
    # extract data
    FAST_sim = sim_detail.FAST_sim


    nseeds = model_data['nseeds']
    W = model_data['W']
    n_var = model_data['n_var']
   
    nW = len(W)


    FAST_sim_array = np.array(FAST_sim)
    FAST_sim = np.reshape(FAST_sim_array,[nseeds,nW],order = 'F')


    train_inds = np.arange(0,5)
    

    n_samples = 1

    w_start = model_data['w_start']

    ind_start = np.where(W == w_start)[0][0]
    ind_forward_pass = np.where(W >= w_start)[0]
    ind_backward_pass = np.flip(np.where(W < w_start))[0]

    A_array = np.zeros((nW,ns*2,ns*2))
    B_array = np.zeros((nW,2*ns,nc))
    C_array = np.zeros((nW,ny,2*ns))
    D_array = np.zeros((nW,ny,nc))

    X_array = np.zeros((nW,n_var))

    model_construct_time = np.zeros((nW,))

    # construct surrogate model
    dfsm_matlab = DFSM(sim_detail,n_samples = n_samples,L_type = 'LPV',N_type = None, train_split = 0.5)
    dfsm_matlab.construct_surrogate()
    dfsm_matlab.simulation_time = []
    dfsm_matlab.train_data = []
    dfsm_matlab.test_data = []

    test_inds = model_data['test_inds']

    # Start the MATLAB Engine
    eng = matlab.engine.start_matlab()

    X_matlab = np.array((nW),dtype = object)
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

        
        inputs_ = matlab.double(inputs)
        state_dx_ = matlab.double(state_dx)
        outputs_ = matlab.double(outputs)

        n_var = matlab.double(n_var)
        ns2 = matlab.double(2*ns)
        nc_ = matlab.double(nc)
        ny_ = matlab.double(ny)

        
        if W[iw] == w_start:

            x0 = matlab.double([])
            
            t1 = timer.time()
            A,B,C,D,x,x_cd = eng.construct_LPV(x0,x0,n_var,ns2,nc_,ny_,inputs_,state_dx_,outputs_,model_data['gaopt'],model_data['fminconopt'],nargout = 6)
            t2 = timer.time()
                
        else:

            x0 = matlab.double(X_array[iw-1])
            x0_cd = matlab.double(X_cd_array[iw-1])

            t1 = timer.time()
            A,B,C,D,x,x_cd = eng.construct_LPV(x0,x0_cd,n_var,ns2,nc_,ny_,inputs_,state_dx_,outputs_,model_data['gaopt'],model_data['fminconopt'],nargout = 6)
            t2 = timer.time()

            
        # stop timer
        model_construct_time[iw] = t2-t1

        # store soultion
        X_array[iw,:] = np.squeeze(np.array(x))
        X_cd_array[iw] = x_cd

        # convert the A,B,C,D matrices to np arrays
        A = np.array(A);B = np.array(B)
        C = np.array(C); D = np.array(D)


        C[3,:] = A[3,:]
        D[3,:] = B[3,:]

        # store linear models
        A_array[iw,:,:] = A
        B_array[iw,:,:] = B
        C_array[iw,:,:] = C
        D_array[iw,:,:] = D

    # perform backward pass
    for ind in range(len(ind_backward_pass)):

        iw = ind_backward_pass[ind]

        _,_,_,inputs,state_dx,outputs = sample_data(FAST_sim[train_inds,iw],'KM',n_samples = 1)

        inputs_ = matlab.double(inputs)
        state_dx_ = matlab.double(state_dx)
        outputs_ = matlab.double(outputs)


        # calculate the C and D matrices
        CD = lstsq(inputs,outputs,rcond = None)
        CD = CD[0]

        CD = CD.T

        C_ = CD[:,nc:]
        D_ = CD[:,:nc]

        C_array[iw,:,:] = C
        D_array[iw,:,:] = D

        x0 = matlab.double(X_array[iw+1])
        x0_cd = X_cd_array[iw+1]
        

        t1 = timer.time()
        A,B,C,D,x,x_cd = eng.construct_LPV(x0,x0_cd,n_var,ns2,nc_,ny_,inputs_,state_dx_,outputs_,model_data['gaopt'],model_data['fminconopt'],nargout = 6)
        t2 = timer.time()

        model_construct_time[iw] = t2-t1

        # store soultion
        X_array[iw,:] = np.squeeze(np.array(x))
        X_cd_array[iw] = x_cd

        A = np.array(A);B = np.array(B)
        C = np.array(C); D = np.array(D);

        C[3,:] = A[3,:]
        D[3,:] = B[3,:]

        # store linear models
        A_array[iw,:,:] = A
        B_array[iw,:,:] = B
        C_array[iw,:,:] = C
        D_array[iw,:,:] = D

    print('shutting matlab down')
    eng.quit()

    A0 = np.squeeze(A_array[0,:,:]);A1 = np.squeeze(A_array[-1,:,:])
    B0 = np.squeeze(B_array[0,:,:]);B1 = np.squeeze(B_array[-1,:,:])
    C0 = np.squeeze(C_array[0,:,:]);C1 = np.squeeze(C_array[-1,:,:])
    D0 = np.squeeze(D_array[0,:,:]);D1 = np.squeeze(D_array[-1,:,:])

    interp_type = 'linear'

    # construct surrogate model
    dfsm_python = DFSM(sim_detail,n_samples = n_samples,L_type = 'LPV',N_type = None, train_split = 0.5)

    dfsm_python.A_array = A_array
    dfsm_python.B_array = B_array
    dfsm_python.C_array = C_array
    dfsm_python.D_array = D_array
    dfsm_python.W = W

    dfsm_python.setup_LPV(interp_type)

    dfsm_python.AB = []
    dfsm_python.CD = []

    dfsm_python.error_ind_deriv = []
    dfsm_python.error_ind_outputs = []

    dfsm_python.dx_error = []
    dfsm_python.test_data = []
    dfsm_python.train_data =[]

    dfsm_python.nonlin_deriv = []
    dfsm_python.nonlin_outputs = []
    dfsm_python.nonlin_output = False
    
    
    dfsm_python.scaler_dx = None
    dfsm_python.scaler_outputs = None
    dfsm_python.nonlin_deriv = np.array([None])
    dfsm_python.nonlin_outputs = np.array([None])



    # load results from matlab
    save_flag = True;save_dict_flag = True;plot_flag = False
    plot_path = 'LPV_results_comp'

    
    # test dfsm
    # flags related to testing
    simulation_flag = True 
    outputs_flag = (len(reqd_outputs) > 0)
    plot_flag = True
    #dfsm,U_list,X_list_matlab,dx_list,Y_list_matlab = test_dfsm(dfsm_matlab,FAST_sim_array,test_inds,simulation_flag,plot_flag)
    dfsm,U_list,X_list_python,dx_list,Y_list_python = test_dfsm(dfsm_python,FAST_sim_array,test_inds,simulation_flag,plot_flag)
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





    plt.show()

    dfsm_python_pkl = 'dfsm_1p6_python.pkl'

    with open(dfsm_python_pkl,'wb') as handle:
        pickle.dump(dfsm_python,handle)

    results_dict = {'U_list':U_list,'X_list':X_list_python,'dx_list':dx_list,'Y_list':Y_list_python,'DFSM':dfsm,'current_speed':current_speed}





        