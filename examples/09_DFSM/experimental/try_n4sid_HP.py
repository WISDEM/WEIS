import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle

from scipy.interpolate import CubicSpline,interp1d


import time as timer

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension,calculate_MSE
from weis.dfsm.dfsm_sample_data import sample_data
from rosco.toolbox import control_interface as ROSCO_ci
from rosco import discon_lib_path

import matlab.engine


plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 15

# get path to this directory
this_dir = os.path.dirname(os.path.realpath(__file__))
outputs_dir = this_dir + os.sep + 'outputs'

def ModelData():
    
    model_data = {}

    model_data['reqd_states'] = ['PtfmSurge','PtfmPitch','TTDspFA','GenSpeed'];ns = len(model_data['reqd_states'])
    model_data['reqd_controls'] =  ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev'];nc = len(model_data['reqd_controls'])
    model_data['reqd_outputs'] = ['TwrBsFxt','TwrBsMyt','NcIMURAys','GenPwr','GenSpeed','PtfmPitch','TTDspFA','PtfmSurge'];ny = len(model_data['reqd_outputs']) 

    model_data['datapath'] = outputs_dir+os.sep +'train_1p62'
    model_data['test_datapath'] = outputs_dir+os.sep +'test_1p62'


    scale_args = {'state_scaling_factor': np.array([1,1,1,1]),
                  'control_scaling_factor': np.array([1,1000,1,1]),
                  'output_scaling_factor': np.ones((ny,))
                  }
    
    model_data['scale_args'] = scale_args

    model_data['nx'] = 2

    # current speed to start the optimization
    model_data['w_test'] = 18

    # indices to be used for training and testing
    model_data['test_inds'] = np.arange(0,10)
    
    # training indices
    model_data['train_inds'] = np.arange(0,5)

    # pickle file to save the DFSM to
    model_data['dfsm_file_name'] = 'fowt_1p6_n4sid.pkl'

    # Use simulations from tmin seconds
    model_data['tmin'] = 200

    model_data['dt_extract'] = 0.01
     
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


    return A,B,C,x0,model_construct_time
    


if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # datapath
    region = 'LPV'

    model_data = ModelData()
    datapath =  model_data['datapath'] #this_dir + os.sep + 'outputs' + os.sep + 'FOWT_1p6' #+ os.sep + 'openfast_runs/rank_0'
    test_datapath =  model_data['test_datapath']
    
    # get the path to all .outb files in the directory
    outfiles_train = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles_train = sorted(outfiles_train)

    # get the path to all .outb files in the directory
    outfiles_test = [os.path.join(test_datapath,f) for f in os.listdir(test_datapath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)

    # required states
    reqd_states = model_data['reqd_states'] 
    reqd_controls = model_data['reqd_controls']
    reqd_outputs = model_data['reqd_outputs']

    ns = len(reqd_states)
    nc = len(reqd_controls)
    ny = len(reqd_outputs)

    gs_ind = reqd_outputs.index('GenSpeed')
    faacc_ind = reqd_outputs.index('NcIMURAys')
    blade_pitch_ind = reqd_controls.index('BldPitch1')
    gen_tq_ind = reqd_controls.index('GenTq')
    ty_ind = reqd_outputs.index('TwrBsMyt')

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
    tmin = model_data['tmin']
    sim_detail_train = SimulationDetails(outfiles_train, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=tmin
                                   ,add_dx2 = True,linear_model_file = None,region = region)
    
    sim_detail_test = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=tmin
                                   ,add_dx2 = True,linear_model_file = None,region = region)
    
    # load and process data
    sim_detail_train.load_openfast_sim()
    sim_detail_test.load_openfast_sim()

    # extract data
    FAST_sim_train = sim_detail_train.FAST_sim
    FAST_sim_array = np.array(FAST_sim_train)

    FAST_sim_test = sim_detail_test.FAST_sim
    FAST_sim_test = np.array(FAST_sim_test)


    w_array = []

    for sim_det in FAST_sim_array:
        w_array.append(sim_det['w_mean'])

    w_array = np.round(np.array(w_array))
    W = np.unique(w_array)
    
    nW = len(W)
    nseeds = int(len(w_array)/nW)
    
    FAST_sim_train = np.reshape(FAST_sim_array,[nseeds,nW],order = 'F')
    FAST_sim_test = np.reshape(FAST_sim_test,[10,nW],order = 'F')
    
    w_array = np.reshape(w_array,[nseeds,nW],order = 'F')

    sort_ind = np.argsort(w_array[0,:])
    FAST_sim_train = FAST_sim_train[:,sort_ind]

    ## Control type
    ctrl_type = 'CL'

    # path to DISCON library
    lib_name = discon_lib_path

    # Write parameter input file
    param_filename = datapath + os.sep + 'DLC1.6_0_weis_job_00_DISCON.IN'
    

    n_samples = 1

    w_test_list = [14]
    nx_list = [10]#np.arange(8,14,2)

    n_nx = len(nx_list)
    n_w = len(w_test_list)

    train_inds = model_data['train_inds']
    test_inds = model_data['test_inds']

    nt = len(FAST_sim_train[0,0]['controls'])

    n_tests = len(train_inds)
    GS_list = []
    BP_list = []

    for nx in nx_list:
        print('--------------------------')
        print(nx)
        print('--------------------------')

        GS_mse = np.zeros(n_w)
        BP_mse = np.zeros(n_w)
        TY_mse = np.zeros(n_w)

        plot_path = 'sys_id' + os.sep +'nx_'+str(int(nx))

        for iw,w_test in enumerate(w_test_list):

            print(w_test)
        
            # wind speed
            ind_w = np.where(W == w_test)[0][0]

            # extract the corresponding inputs and outputs
            _,_,_,inputs,state_dx,outputs = sample_data(FAST_sim_train[train_inds,ind_w],'KM',n_samples = 1)
            inputs = np.array(inputs[:,:nc])

            # get scaler for outputs
            outputs_max = np.abs(np.max(outputs,0))
            max_ind = outputs_max < 1e-4
            outputs_max[max_ind] = 1 

            # get system matrices by calling matlab
            A,B,C,x0,mc_time = call_matlab(inputs,outputs,nc,ny,outputs_max,n_tests,int(nx),nt,dt_extract)

            # load results from matlab
            save_flag = True;save_dict_flag = True;plot_flag = True
            
            ts_path = plot_path + os.sep + 'w_'+str(int(w_test))

            gs_mse_w = np.zeros(len(test_inds))
            gt_mse_w = np.zeros(len(test_inds))
            bp_mse_w = np.zeros(len(test_inds))
            ty_mse_w = np.zeros(len(test_inds))
            me_time = np.zeros(len(test_inds))

            for idx,ind in enumerate(test_inds):
                
                FS = FAST_sim_test[ind,ind_w]

                
                U_of = FS['controls']
                outputs_OF = FS['outputs']

                U_n4sid = np.zeros(U_of.shape)

                U_n4sid[0,:] = U_of[0,:]

                #time = results_mat['time']
                time = FAST_sim_test[ind,ind_w]['time']
                time = time-200
                
                tf = time[-1]

                nt = len(time)

                # initalize
                X = np.zeros((nt+1,nx))
                Y = np.zeros((nt,ny))
                
                X[0,:] = x0[:,1]
                t1 = timer.time()
                if ctrl_type == 'CL':
                    # parameters for ROSCO
                    num_blade = int(3)

                    bp0 = U_of[0,blade_pitch_ind]
                    args = {'DT': dt_extract, 'num_blade': int(3),'pitch':bp0}

                    rpm2RadSec = 2.0*(np.pi)/60.0
                    KWatt2Watt = 1000
                    Deg2Rad = np.pi/180

                    controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)

                    wind_fun = CubicSpline(time,U_of[:,0])
                    wave_fun = CubicSpline(time,U_of[:,-1])

                elif ctrl_type == 'OL':
                    U_n4sid = U_of

                for i in range(nt):

                    if ctrl_type == 'OL':
                        u_ = U_n4sid[i,:]

                    elif ctrl_type == 'CL':

                        if i == 0:
                            u_ = U_n4sid[i,:]

                        else:

                            t = time[i]
                            turbine_state = {}

                            if t == tf:
                                turbine_state['iStatus']    = -1
                            else:
                                turbine_state['iStatus']    = 1
                            
                            turbine_state['bld_pitch'] = np.deg2rad(U_n4sid[i-1,blade_pitch_ind]) # blade pitch
                            turbine_state['gen_torque'] = U_n4sid[i-1,gen_tq_ind]*KWatt2Watt*1000 # generator torque
                            turbine_state['t'] = t # current time step
                            turbine_state['dt'] = dt_extract # step size
                            turbine_state['ws'] = U_of[i,0] # estimate wind speed
                            turbine_state['num_blades'] = int(3) # number of blades
                            turbine_state['gen_speed'] = Y[i-1,gs_ind]*rpm2RadSec*1 # generator speed
                            turbine_state['gen_eff'] = 95.89835000000/100 # generator efficiency
                            turbine_state['rot_speed'] = Y[i-1,gs_ind]*rpm2RadSec*1# rotor speed
                            turbine_state['Yaw_fromNorth'] = 0 # yaw
                            turbine_state['Y_MeasErr'] = 0
                            turbine_state['NacIMU_FA_Acc'] = Y[i-1,faacc_ind]*np.deg2rad(1)

                            gen_torque, bld_pitch, nac_yawrate = controller_interface.call_controller(turbine_state)

                            U_n4sid[i,0] = wind_fun(t)
                            U_n4sid[i,gen_tq_ind] = gen_torque/(KWatt2Watt*1000)
                            U_n4sid[i,blade_pitch_ind] = np.rad2deg(bld_pitch)
                            U_n4sid[i,-1] = wave_fun(t)

                            u_ = U_n4sid[i,:]

                    x = X[i,:]

                    xi = np.dot(A,x) + np.dot(B,u_)
                    
                    Y[i,:] = np.dot(C,x)*outputs_max
                    X[i+1,:] = xi
                t2 = timer.time()
                

                if ctrl_type == 'CL':
                    controller_interface.kill_discon()


                    gt_mse_w[idx] = calculate_MSE(U_of[:,blade_pitch_ind],U_n4sid[:,blade_pitch_ind])
                    bp_mse_w[idx] = calculate_MSE(U_of[:,gen_tq_ind],U_n4sid[:,gen_tq_ind])
                print(t2-t1)

                gs_mse_w[idx] = calculate_MSE(outputs_OF[:,gs_ind],Y[:,gs_ind])
                ty_mse_w[idx] = calculate_MSE(outputs_OF[:,ty_ind],Y[:,ty_ind])
                me_time[idx] = t2-t1
                


                
                if plot_flag:


                    fig,ax = plt.subplots(1)
                    ax.plot(time,U_of[:,0])
                    ax.tick_params(labelsize=fontsize_tick)
                    ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                    ax.set_title('RtVAvgxh',fontsize = fontsize_axlabel)

                    if save_flag:
                        if not os.path.exists(ts_path):
                            os.makedirs(ts_path)
                
                        fig.savefig(ts_path +os.sep+ 'RtVAvgxh_' +str(idx)+ '.pdf')
                    plt.close(fig)

                    if ctrl_type == 'CL':

                        for iu in [1,2]:

                            if reqd_controls[iu] == 'BldPitch1':
                                title = 'Blade Pitch [deg]'
                            else:
                                title = reqd_controls[iu]


                            fig,ax = plt.subplots(1)
                            ax.plot(time,U_of[:,iu],color = 'k',label = 'OpenFAST')
                            ax.plot(time,U_n4sid[:,iu],color = 'tab:orange',label = 'n4sid')
                            ax.tick_params(labelsize=fontsize_tick)
                            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                            ax.set_title(title,fontsize = fontsize_axlabel)
                            ax.set_xlim([0,600])

                            if save_flag:
                                if not os.path.exists(ts_path):
                                    os.makedirs(ts_path)
                        
                                fig.savefig(ts_path +os.sep+ reqd_controls[iu]+'_' +str(idx)+ '.pdf')
                            plt.close(fig)

                    for i in range(ny):

                        if reqd_outputs[i] == 'GenSpeed':
                            title = 'Generator Speed [rpm]'
                        else:
                            title = reqd_outputs[i]

                        if reqd_outputs[i] == 'TwrBsMyt':
                            title = 'Tower-Base Moment [kNm]'
                        else:
                            title = reqd_outputs[i]

                        fig,ax1 = plt.subplots(1)
                        
                        ax1.plot(time,outputs_OF[:,i],color = 'k',label = 'OpenFAST')
                        ax1.plot(time,Y[:,i],color = 'tab:orange',label = 'n4sid')
                        ax1.set_xlim([0,600])
                        
                        ax1.tick_params(labelsize=fontsize_tick)
                        ax1.legend(ncol = 2,fontsize = fontsize_legend)
                        ax1.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                        ax1.set_title(title,fontsize = fontsize_axlabel)

                        if save_flag:
                            if not os.path.exists(ts_path):
                                os.makedirs(ts_path)
                    
                            fig.savefig(ts_path +os.sep+ reqd_outputs[i] +'_' +str(idx)+ '_comp.pdf')
                        plt.close(fig)

            GS_mse[iw] = np.mean(gs_mse_w)
            TY_mse[iw] = np.mean(ty_mse_w)
            BP_mse[iw] = np.mean(bp_mse_w)

            n4sid_model = {'A':A,'B':B,'C':C,'x0':x0,'mc_time':mc_time,'outputs_max':outputs_max}

            with open(ts_path + os.sep +'n4sid_model.pkl','wb') as handle:
                pickle.dump(n4sid_model,handle)

            results_dict = {'bp_mse':bp_mse_w,'gs_mse':gs_mse_w,'twrbsmyt_mse':ty_mse_w,'model_sim_time':me_time}
            results_file = ts_path+os.sep+'results_dict.pkl'

            with open(results_file,'wb') as handle:
                pickle.dump(results_dict,handle)

        fig,ax = plt.subplots(1)

        ax.set_title('nx = '+str(int(nx)))
        ax.plot(w_test_list,GS_mse,'.-',markersize = markersize,label = 'GS')
        ax.tick_params(labelsize=fontsize_tick)
        ax.set_yscale('log')
        ax.legend(ncol = 2,fontsize = fontsize_legend)
        ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)

        if save_flag:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
    
            fig.savefig(plot_path +os.sep+ 'MSE_W_GS.pdf')
        plt.close(fig)

        if ctrl_type == 'CL':

            BP_list.append(BP_mse)
            fig,ax = plt.subplots(1)

            ax.set_title('nx = '+str(int(nx)))
            ax.plot(w_test_list,BP_mse,'.-',markersize = markersize,label = 'BP')
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_yscale('log')
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)

            if save_flag:
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
        
                fig.savefig(plot_path +os.sep+ 'MSE_W_BP.pdf')
            plt.close(fig)


        GS_list.append(GS_mse)

        
    GS_mse_array = np.array(GS_list)
    
    fig,ax = plt.subplots(1)

    ax.set_title('MSE GS')
    for i in range(len(nx_list)):
        ax.plot(w_test_list,GS_mse_array[i,:],'.-',markersize = markersize,label = 'nx = '+str(int(nx_list[i])))
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
    ax.set_yscale('log')
    ax.legend(ncol = 2,fontsize = fontsize_legend)
    if save_flag:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        fig.savefig('sys_id_test2' +os.sep+ 'MSE_GS_all.pdf')
    plt.close(fig)

    if ctrl_type == 'CL':

        BP_mse_array = np.array(BP_list)
    
        fig,ax = plt.subplots(1)

        ax.set_title('MSE BP')
        for i in range(len(nx_list)):
            ax.plot(w_test_list,BP_mse_array[i,:],'.-',markersize = markersize,label = 'nx = '+str(int(nx_list[i])))
        ax.tick_params(labelsize=fontsize_tick)
        ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
        ax.set_yscale('log')
        ax.legend(ncol = 2,fontsize = fontsize_legend)
        if save_flag:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            fig.savefig('sys_id_test2' +os.sep+ 'MSE_BP_all.pdf')
        plt.close(fig)

    breakpoint()