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
from keras.layers import LSTM,Dense,GRU
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tqdm import tqdm

# path to this directory
this_dir = os.path.dirname(os.path.abspath(__file__))

exp_name = 'LSTM_14_test'

RNN_save_fol = this_dir + os.sep +'RNN_models' + os.sep + exp_name
if not os.path.exists(RNN_save_fol):
    os.makedirs(RNN_save_fol)


def model_params():

    model_params = {}

    model_params['w_list'] = [14]
    model_params['w_test'] = 14

    model_params['n_exp'] = 5
    model_params['ind_test_list'] = [0,1,2,3,4,5,6,7,8,9]

    model_params['model_save_name'] = RNN_save_fol + os.sep  + 'FOWT_1p6_lstm_14.keras'

    model_params['ctrl_type'] = 'CL'
    model_params['save_flag'] = True
    model_params['plot_path'] = RNN_save_fol + os.sep +'plots_validations'

    model_params['dt_extract'] = 0.5
    model_params['RNN_type'] = 'LSTM'
    model_params['n_cells'] = 10

    model_params['n_epoch'] = 10

    model_params['dt_list'] = [0.5]


    return model_params



if __name__ == '__main__':

    
    # path to DISCON library
    lib_name = discon_lib_path

    # Write parameter input file
    param_filename = '/home/athulsun/DEV/examples/19_DFSM/validation_test/weis_job_0_DISCON.IN'
    testpath = '/home/athulsun/DEV/examples/19_DFSM/validation_test'
    

    outfiles_test = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)
    
    # required states
    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    
    state_props = {'units' : ['[deg]','[m/s2]','[rpm]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = ['GenSpeed','NcIMURAys','TwrBsMyt','PtfmPitch','TTDspFA'] 
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,1,100]),
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
    
    # call model_params
    model_params = model_params()
    
    sim_details = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = False,linear_model_file = [],dt_extract =model_params['dt_extract'])
    
    sim_details.load_openfast_sim()

    FAST_sim,w_unique,n_unique = reorganize_data(sim_details.FAST_sim)
    n_cases = FAST_sim.shape[0]
    

    w_list = model_params['w_list']
    FAST_sim_ind = np.zeros((n_cases,len(w_list)),dtype = object)

    model_save_name = model_params['model_save_name']

    model_exists = os.path.exists(model_save_name)

    nc = len(reqd_controls)
    ny = len(reqd_outputs)

    if len(w_list) == 1:

        FAST_sim_ind = FAST_sim

    else:

        for iw,w_ in enumerate(w_list):
            ind_w = w_unique == w_ 
            FAST_sim_ind[:,iw] = np.squeeze(FAST_sim[:,ind_w])

        FAST_sim_ind = np.array(FAST_sim_ind)
        FAST_sim_ind = np.reshape(FAST_sim_ind,[n_cases,iw+1],order = 'C')

    w_test_ind = w_list.index(model_params['w_test'])

    if not model_exists:
        n_exp = model_params['n_exp']
        FAST_sim_ind_train = FAST_sim_ind[:n_exp,:]

        FAST_sim_ind_train = np.reshape(FAST_sim_ind_train,[(iw+1)*n_exp,],order = 'F')
        _,_,_,model_inputs,state_derivatives,outputs_train = sample_data(FAST_sim_ind_train,sampling_type = 'KM',n_samples = 2)

        controls_train = model_inputs[:,:nc]
        n_pts = len(controls_train);print(n_pts)


    scaler_control_file = RNN_save_fol + os.sep  +'scaler_controls.pkl'
    scaler_outputs_file = RNN_save_fol + os.sep  +'scaler_outputs.pkl'

    if os.path.exists(scaler_control_file) and os.path.exists(scaler_outputs_file):

        with open(scaler_control_file,'rb') as handle:
            scaler_controls = pickle.load(handle)

        with open(scaler_outputs_file,'rb') as handle:
            scaler_outputs = pickle.load(handle)
    
    else:

        scaler_controls = MinMaxScaler()
        scaler_outputs = MinMaxScaler()

        scaler_controls.fit(controls_train)

        controls_train_sc = scaler_controls.transform(controls_train)

        scaler_outputs.fit(outputs_train)

        outputs_train_sc = scaler_outputs.transform(outputs_train)

        with open(scaler_control_file ,'wb') as handle:
            pickle.dump(scaler_controls,handle)

        with open(scaler_outputs_file ,'wb') as handle:
            pickle.dump(scaler_outputs,handle)

    if not model_exists:
        n_batch = n_pts
        
        controls_train_sc = np.reshape(controls_train_sc,[n_batch,1,nc],order = 'F')
        outputs_train_sc = np.reshape(outputs_train_sc,[n_batch,1,ny],order = 'F')


    t1 = timer.time()
    if model_exists:
        model = load_model(model_save_name)
    else:
        # 
        
        model = tf.keras.Sequential()
        n_cells = model_params['n_cells']

        if model_params['RNN_type'] == 'LSTM':

            model.add(tf.keras.layers.LSTM(n_cells))

        elif model_params['RNN_type'] == 'GRU':

            model.add(tf.keras.layers.GRU(n_cells))

        model.add(tf.keras.layers.Dense(ny))
        model.compile(optimizer = 'adam',loss = 'mse')
        model.fit(controls_train_sc,outputs_train_sc,batch_size = 1,epochs = model_params['n_epoch'],verbose = 2)

        model.save(model_save_name)
    t2 = timer.time()
    print(t2-t1)                    

    ind_test_list  = model_params['ind_test_list'] 
    dt_list = model_params['dt_list']
    n_dt = len(dt_list)

    n_test = len(ind_test_list)
    bp_mse = np.zeros((n_test,))
    twrbsmyt_mse = np.zeros((n_test,))
    gs_mse = np.zeros((n_test,))
    model_sim_time = np.zeros((n_test,))


    results_dict = {}

    ctrl_type = model_params['ctrl_type']
    save_flag = model_params['save_flag']
    plot_path = model_params['plot_path']

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 18
    fontsize_tick = 15
    t_transition = 100

    ylim_list = [[5,10],[-1,1],[-2e5,4e5],[-1,6],[-0.8,0.8]]

    for i_test,ind_test in enumerate(ind_test_list):
        

        controls_test = FAST_sim_ind[ind_test,w_test_ind]['controls']
        outputs_test = FAST_sim_ind[ind_test,w_test_ind]['outputs']
        time = FAST_sim_ind[ind_test,w_test_ind]['time']

        t_ind = time >=t_transition

        t0 = time[0];tf = time[-1]

        genspeed_ind = reqd_outputs.index('GenSpeed')
        fa_acc_ind = reqd_outputs.index('NcIMURAys')
        gen_torque_ind = reqd_controls.index('GenTq')
        blade_pitch_ind = reqd_controls.index('BldPitch1')

        nt = len(time)
        nu  = len(reqd_controls)
        ny = len(reqd_outputs)

        if ctrl_type == 'OL':

            controls_test_sc = scaler_controls.transform(controls_test)

            time_test = time

            t1 = timer.time()
            outputs_test_pred = model.predict(controls_test_sc)
            t2 = timer.time()

            outputs_test_pred = scaler_outputs.inverse_transform(outputs_test_pred)

        elif ctrl_type == 'CL':

            
            for idt,dt_ in enumerate(dt_list):
            
                time_test = np.arange(t0,tf+dt_,dt_)
                nt_ = len(time_test)

                outputs_test_pred = np.zeros((nt_,ny))
                controls_test_pred = np.zeros((nt_,nu))

                controls_test_pred[0,:] = controls_test[0,:]
                outputs_test_pred[0,:] = outputs_test[0,:]

                wind_fun = CubicSpline(time,controls_test[:,0])
                wave_fun = CubicSpline(time,controls_test[:,-1])

                # parameters for ROSCO
                num_blade = int(3)

                bp0 = controls_test[0,blade_pitch_ind]
                args = {'DT': dt_, 'num_blade': int(3),'pitch':bp0}

                rpm2RadSec = 2.0*(np.pi)/60.0
                KWatt2Watt = 1000
                Deg2Rad = np.pi/180

                controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)
                t1 = timer.time()
                for i in tqdm(range(nt_)):

                    if i == 0:
                        continue
                    else:

                        t = time_test[i]
                    
                        turbine_state = {}

                        if t == tf:
                            turbine_state['iStatus']    = -1
                        else:
                            turbine_state['iStatus']    = 1

                        turbine_state['bld_pitch'] = np.deg2rad(controls_test_pred[i-1,blade_pitch_ind]) # blade pitch
                        turbine_state['gen_torque'] = controls_test_pred[i-1,gen_torque_ind]*KWatt2Watt # generator torque
                        turbine_state['t'] = t # current time step
                        turbine_state['dt'] = dt_ # step size
                        turbine_state['ws'] = controls_test_pred[i-1,0] # estimate wind speed
                        turbine_state['num_blades'] = int(3) # number of blades
                        turbine_state['gen_speed'] = outputs_test_pred[i-1,genspeed_ind]*rpm2RadSec*1 # generator speed
                        turbine_state['gen_eff'] = 95.89835000000/100 # generator efficiency
                        turbine_state['rot_speed'] = outputs_test_pred[i-1,genspeed_ind]*rpm2RadSec*1# rotor speed
                        turbine_state['Yaw_fromNorth'] = 0 # yaw
                        turbine_state['Y_MeasErr'] = 0
                        turbine_state['NacIMU_FA_Acc'] = outputs_test_pred[i-1,fa_acc_ind]*np.deg2rad(1)

                        gen_torque, bld_pitch, nac_yawrate = controller_interface.call_controller(turbine_state)

                        controls_test_pred[i,0] = wind_fun(t)
                        controls_test_pred[i,gen_torque_ind] = gen_torque/KWatt2Watt
                        controls_test_pred[i,blade_pitch_ind] = np.rad2deg(bld_pitch)
                        controls_test_pred[i,-1] = wave_fun(t)

                        ci = controls_test_pred[i,:]


                        controls_test_sc = scaler_controls.transform(ci.reshape(1,-1))
                        controls_test_sc = controls_test_sc.reshape([1,1,-1])
                        
                        #print(t)
                        o_pred = scaler_outputs.inverse_transform(model.predict(controls_test_sc,verbose = 0))

                        outputs_test_pred[i,:] = np.squeeze(o_pred)
                t2 = timer.time()
                model_sim_time[i_test] = t2-t1
                print(t2-t1)
                controller_interface.kill_discon()
                time[t_ind] = time[t_ind]-100
                #time_test[t_ind] = time_test[t_ind]-100

      
                for i_out,output in enumerate(reqd_outputs):

                    fig,ax = plt.subplots(1)

                    ax.plot(time[t_ind],outputs_test[t_ind,i_out],label = 'OpenFAST')
                    ax.plot(time_test[t_ind]-100,outputs_test_pred[t_ind,i_out],label = 'LSTM')

                    ax.set_title(output,fontsize = fontsize_axlabel)
                    ax.set_ylim(ylim_list[i_out])
                    ax.tick_params(labelsize=fontsize_tick)
                    ax.legend(ncol = 2,fontsize = fontsize_legend)
                    ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                    ax.set_xlim([0,tf-t_transition])


                    if output == 'GenSpeed':
                        gs_of = outputs_test[t_ind,i_out]
                        gs_dfsm = CubicSpline(time_test,outputs_test_pred[:,i_out])
                        gs_dfsm = gs_dfsm(time[t_ind])

                        gs_mse[i_test] = calculate_mse(gs_of,gs_dfsm)

                    if output == 'TwrBsMyt':
                        M_of = outputs_test[t_ind,i_out]
                        M_dfsm = CubicSpline(time_test,outputs_test_pred[:,i_out])
                        M_dfsm = M_dfsm(time[t_ind])

                        twrbsmyt_mse[i_test] = calculate_mse(M_of,M_dfsm)

                    if save_flag:
                        if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                
                        fig.savefig(plot_path +os.sep+ output +'_' +str(i_test)+ '_comp.pdf')
                        plt.close(fig)

                if ctrl_type == 'CL':

                    for i_out,control in enumerate(reqd_controls):

                        fig,ax = plt.subplots(1)

                        ax.plot(time[t_ind],controls_test[t_ind,i_out],color = 'k',label = 'OpenFAST')
                        ax.plot(time_test[t_ind]-100,controls_test_pred[t_ind,i_out],color = 'b',label = 'LSTM')

                        ax.set_title(control,fontsize = fontsize_axlabel)
                        ax.tick_params(labelsize=fontsize_tick)
                        ax.legend(ncol = 2,fontsize = fontsize_legend)
                        ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)

                        ax.set_xlim([0,tf-t_transition])
 

                        if control == 'BldPitch1':
                            bp_of = controls_test[t_ind,i_out]
                            bp_dfsm = CubicSpline(time_test,controls_test_pred[:,i_out])
                            bp_dfsm = bp_dfsm(time[t_ind])

                            bp_mse[i_test] = calculate_mse(bp_of,bp_dfsm)


                        if save_flag:
                            if not os.path.exists(plot_path):
                                os.makedirs(plot_path)
                    
                            fig.savefig(plot_path +os.sep+ control +'_' +str(i_test)+ '_comp.pdf')
                            plt.close(fig)

    results_dict = {'bp_mse':bp_mse,'gs_mse':gs_mse,'twrbsmyt_mse':twrbsmyt_mse,'model_sim_time':model_sim_time}
    results_file = plot_path+os.sep+'results_dict.pkl'
    
    with open(results_file,'wb') as handle:
        pickle.dump(results_dict,handle)

    plt.show()

    