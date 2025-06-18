import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os
import time as timer
from scipy.interpolate import CubicSpline

# ROSCO toolbox modules 
from rosco.toolbox import control_interface as ROSCO_ci
from rosco import discon_lib_path
from rosco.toolbox.inputs.validation import load_rosco_yaml
from rosco.toolbox.utilities import write_DISCON
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import controller as ROSCO_controller

# DFSM modules
from weis.dfsm.ode_algorithms import RK4
from weis.dfsm.dfsm_utilities import calculate_MSE as calculate_mse
from scipy.interpolate import interp1d

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension,reorganize_data
from weis.dfsm.construct_dfsm import DFSM
from weis.dfsm.dfsm_sample_data import sample_data

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tqdm import tqdm

def create_model(n_neuron,batch_size,ny,shape_):
    model = Sequential()
    model.add(LSTM(n_neuron, input_shape=(batch_size, shape_[1], shape_[2]), stateful=True))
    model.add(Dense(ny))
    return model


def scale_and_predict(yi,xi,scaler_outputs,scaler_controls,model):

    yi = scaler_outputs.transform(yi.reshape(1,-1))
    xi = scaler_controls.transform(xi.reshape(1,-1))

    model_inputs_iter = np.hstack([yi,xi])
    model_inputs_iter = model_inputs_iter.reshape([1,1,-1])

    yi1 = scaler_outputs.inverse_transform(model.predict(model_inputs_iter,verbose = 0))

    return yi1


if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # path to DISCON library
    lib_name = discon_lib_path

    # Write parameter input file
    param_filename = '/home/athulsun/DFSM/data/FOWT_1p6/weis_job_00_DISCON.IN'

    testpath = '/home/athulsun/DFSM/data/FOWT_1p6'
    

    outfiles_test = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)
    
    # required states
    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    
    state_props = {'units' : ['[deg]','[m/s2]','[rpm]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = ['GenSpeed','NcIMURAys','TwrBsMyt'] 
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,1,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.ones((len(reqd_outputs),))
                  }
    
    # filter parameters
    filter_args = {'state_filter_flag': [False,False,False],
                   'state_filter_type': [['filtfilt'],['filtfilt'],['filtfilt']],
                   'state_filter_tf': [[0.1],[0.1],[0.1]],
                   'control_filter_flag': [False,False,False],
                   'control_filter_tf': [0,0,0],
                   'output_filter_flag': []
                   }
    
    
    sim_details = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = False,linear_model_file = [],dt_extract = 0.5)
    
    sim_details.load_openfast_sim()

    FAST_sim,w_unique,n_unique = reorganize_data(sim_details.FAST_sim)
    
    n_cases = FAST_sim.shape[0]
    w = [16]

    # extract the simulation details corresponding to the wind speeds in 'w'
    FAST_sim_ind = np.zeros((n_cases,len(w)),dtype = object)
    
    for iw,w_ in enumerate(w):
        ind_w = w_unique == w_ 
        FAST_sim_ind[:,iw] = np.squeeze(FAST_sim[:,ind_w])

    # organize and reshape
    FAST_sim_ind = np.array(FAST_sim_ind)
    FAST_sim_ind = np.reshape(FAST_sim_ind,[n_cases,iw+1],order = 'C')

    n_exp = 5

    # the indices corresponding to the first 'n_exp' will be used to train the LSTM model

    # extract
    FAST_sim_ind_train = FAST_sim_ind[:n_exp,:]

    # reshape as an 1D array
    FAST_sim_ind_train = np.reshape(FAST_sim_ind_train,[(iw+1)*n_exp,],order = 'F')

    # extract the inputs and outputs as a single array
    _,_,_,model_inputs,state_derivatives,outputs_train = sample_data(FAST_sim_ind_train,sampling_type = 'KM',n_samples = 2)

    
    nc = len(reqd_controls)
    nout = len(reqd_outputs)
    controls_train = model_inputs[:,:nc]
    n_pts = len(controls_train)
    ny = len(reqd_outputs)
    
    
    scaler_controls = MinMaxScaler()
    scaler_outputs = MinMaxScaler()

    scaler_controls.fit(controls_train)

    controls_train_sc = scaler_controls.transform(controls_train)

    scaler_outputs.fit(outputs_train)

    outputs_train_sc = scaler_outputs.transform(outputs_train)

    with open('scaler_controls.pkl','wb') as handle:
        pickle.dump(scaler_controls,handle)

    with open('scaler_outputs.pkl','wb') as handle:
        pickle.dump(scaler_outputs,handle)
    
    X = []
    Y = []

    for i in range(n_pts-1):
        X.append(np.hstack([outputs_train_sc[i,:],controls_train_sc[i,:]]))
        Y.append(outputs_train_sc[i+1,:])
    
    X = np.array(X);Y = np.array(Y)
    X = np.reshape(X,[X.shape[0],1,X.shape[1]])
    Y = np.reshape(Y,[Y.shape[0],1,Y.shape[1]])
    
    

    model_name = 'FOWT_1p6_lstm_'+'R'+ '.keras'


    t1 = timer.time()
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        # 
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units = 128,return_sequences = True))
        model.add(tf.keras.layers.LSTM(units = 128,stateful = True))
        model.add(tf.keras.layers.Dense(ny))
        model.compile(optimizer = 'adam',loss = 'mse')
        model.fit(X,Y,batch_size = 1,epochs = 50,verbose = 2)

        model.save(model_name)
    model.layers[0].reset_states()
    t2 = timer.time()
    print(t2-t1)                    

    #_,_,_,model_inputs,state_derivatives,outputs_test = sample_data(FAST_sim_ind[5],sampling_type = 'KM',n_samples = 2)
    ind_test_list  = [3,5]
    dt_= 0.5
    n_test = len(ind_test_list)

    bp_mse = np.zeros((n_test))
    twbyt_mse = np.zeros((n_test))
    gs_mse = np.zeros((n_test))

    ctrl_type = 'OL'
    save_flag = True
    plot_path = 'LSTM-results-'+ctrl_type

    results_dict = {}

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 18
    fontsize_tick = 12

    ylim_list = [[5,10],[-1,1],[-2e5,4e5],[-1,6],[-0.8,0.8]]

    for i_test,ind_test in enumerate(ind_test_list):

        controls_test = FAST_sim_ind[ind_test,1]['controls']
        outputs_test = FAST_sim_ind[ind_test,1]['outputs']
        time = FAST_sim_ind[ind_test,1]['time']
        t0 = time[0];tf_ = time[-1]

        controls_fun = CubicSpline(time,controls_test)

        genspeed_ind = reqd_outputs.index('GenSpeed')
        fa_acc_ind = reqd_outputs.index('NcIMURAys')
        gen_torque_ind = reqd_controls.index('GenTq')
        blade_pitch_ind = reqd_controls.index('BldPitch1')

        nt = len(time)
        nu  = len(reqd_controls)
        ny = len(reqd_outputs)

        time_test = np.arange(t0,tf_+dt_,dt_)
        nt_ = len(time_test)

        outputs_test_pred = np.zeros((nt_,ny))
        controls_test_pred = np.zeros((nt_,nu))

        controls_test_pred[0,:] = controls_test[0,:]
        outputs_test_pred[0,:] = outputs_test[0,:]


        if ctrl_type == 'CL':

            # parameters for ROSCO
            num_blade = int(3)

            bp0 = controls_test[0,blade_pitch_ind]
            args = {'DT': dt_, 'num_blade': int(3),'pitch':bp0}

            rpm2RadSec = 2.0*(np.pi)/60.0
            KWatt2Watt = 1000
            Deg2Rad = np.pi/180

            controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)

            wind_fun = CubicSpline(time,controls_test[:,0])
            wave_fun = CubicSpline(time,controls_test[:,-1])

        t1 = timer.time()
        for i in tqdm(range(nt_-1)):

            t = time_test[i]

            if i == 0:

                yi = outputs_test_pred[i,:]
                xi = controls_test_pred[i,:]

                yi1 = scale_and_predict(yi,xi,scaler_outputs,scaler_controls,model)

                outputs_test_pred[i+1,:] = np.squeeze(yi1)

            else:

                if ctrl_type == 'CL':
                
                    turbine_state = {}

                    if t == tf_:
                        turbine_state['iStatus']    = -1
                    else:
                        turbine_state['iStatus']    = 1

                    turbine_state['bld_pitch'] = np.deg2rad(controls_test_pred[i-1,blade_pitch_ind]) # blade pitch
                    turbine_state['gen_torque'] = controls_test_pred[i-1,gen_torque_ind]*KWatt2Watt # generator torque
                    turbine_state['t'] = t # current time step
                    turbine_state['dt'] = dt_ # step size
                    turbine_state['ws'] = controls_test_pred[i,0] # estimate wind speed
                    turbine_state['num_blades'] = int(3) # number of blades
                    turbine_state['gen_speed'] = outputs_test_pred[i,genspeed_ind]*rpm2RadSec*1 # generator speed
                    turbine_state['gen_eff'] = 95.89835000000/100 # generator efficiency
                    turbine_state['rot_speed'] = outputs_test_pred[i,genspeed_ind]*rpm2RadSec*1# rotor speed
                    turbine_state['Yaw_fromNorth'] = 0 # yaw
                    turbine_state['Y_MeasErr'] = 0
                    turbine_state['NacIMU_FA_Acc'] = outputs_test_pred[i,fa_acc_ind]*np.deg2rad(1)

                    gen_torque, bld_pitch, nac_yawrate = controller_interface.call_controller(turbine_state)

                    controls_test_pred[i,0] = wind_fun(t)
                    controls_test_pred[i,gen_torque_ind] = gen_torque/KWatt2Watt
                    controls_test_pred[i,blade_pitch_ind] = np.rad2deg(bld_pitch)
                    controls_test_pred[i,-1] = wave_fun(t)

                    ci = controls_test_pred[i,:]

                elif ctrl_type == 'OL':

                    ci = controls_fun(t)
                    controls_test_pred[i,:] = ci
                
                yi = outputs_test_pred[i,:]

                yi1 = scale_and_predict(yi,xi,scaler_outputs,scaler_controls,model)

                outputs_test_pred[i+1,:] = np.squeeze(yi1)
                
        
        model.layers[1].reset_states()
        t2 = timer.time()
        print(t2-t1)

        if ctrl_type == 'CL':
            controller_interface.kill_discon()

        for i_out,output in enumerate(reqd_outputs):

            fig,ax = plt.subplots(1)

            ax.plot(time,outputs_test[:,i_out],label = 'OpenFAST')
            ax.plot(time_test,outputs_test_pred[:,i_out],label = 'LSTM')

            ax.set_title(output,fontsize = fontsize_axlabel)
            ax.set_ylim(ylim_list[i_out])
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            ax.set_xlim([0,tf_])


            if output == 'GenSpeed':
                    gs_of = outputs_test[:,i_out]
                    gs_dfsm = CubicSpline(time_test,outputs_test_pred[:,i_out])
                    gs_dfsm = gs_dfsm(time)

                    gs_mse[i_test] = calculate_mse(gs_of,gs_dfsm)

                    results_dict['time'] = time
                    results_dict['genspeed_of'] = gs_of 
                    results_dict['genspeed_dfsm'] = gs_dfsm

            if output == 'TwrBsMyt':
                    tbm_of = outputs_test[:,i_out]
                    tbm_dfsm = CubicSpline(time_test,outputs_test_pred[:,i_out])
                    tbm_dfsm = tbm_dfsm(time)

                    results_dict['twrbsmyt_of'] = gs_of 
                    results_dict['twrbsmyt_dfsm'] = gs_dfsm

                    twbyt_mse[i_test] = calculate_mse(tbm_of,tbm_dfsm)


            if save_flag:
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
        
                fig.savefig(plot_path +os.sep+ output +'_' +str(i_test)+ '_comp.pdf')
                plt.close(fig)

            

        for i_out,control in enumerate(reqd_controls):

            fig,ax = plt.subplots(1)

            ax.plot(time,controls_test[:,i_out],label = 'OpenFAST')
            ax.plot(time_test,controls_test_pred[:,i_out],label = 'LSTM')

            ax.set_title(control,fontsize = fontsize_axlabel)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            ax.set_xlim([0,600])


            if control == 'BldPitch1':
                bp_of = controls_test[:,i_out]
                bp_dfsm = CubicSpline(time_test,controls_test_pred[:,i_out])
                bp_dfsm = bp_dfsm(time)

                results_dict['bldpitch_of'] = bp_of 
                results_dict['bldpitch_dfsm'] = bp_dfsm

                bp_mse[i_test] = calculate_mse(bp_of,bp_dfsm)



            if save_flag:
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
        
                fig.savefig(plot_path +os.sep+ control +'_' +str(i_test)+ '_comp.pdf')
                plt.close(fig)

   