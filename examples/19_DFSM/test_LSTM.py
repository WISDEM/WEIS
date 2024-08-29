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


if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # path to DISCON library
    lib_name = discon_lib_path

    # Write parameter input file
    param_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_DISCON.IN')

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
    
    
    sim_details = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = False,linear_model_file = [])
    
    sim_details.load_openfast_sim()

    FAST_sim,w_unique,n_unique = reorganize_data(sim_details.FAST_sim)
    n_cases = FAST_sim.shape[0]
    w = [17,19,21]
    FAST_sim_ind = np.zeros((n_cases,len(w)),dtype = object)

    for iw,w_ in enumerate(w):
        ind_w = w_unique == w_ 
        FAST_sim_ind[:,iw] = np.squeeze(FAST_sim[:,ind_w])

    FAST_sim_ind = np.array(FAST_sim_ind)
    FAST_sim_ind = np.reshape(FAST_sim_ind,[n_cases,iw+1],order = 'C')

    n_exp = 5

    FAST_sim_ind_train = FAST_sim_ind[:n_exp,:]

    FAST_sim_ind_train = np.reshape(FAST_sim_ind_train,[(iw+1)*n_exp,],order = 'F')
    _,_,_,model_inputs,state_derivatives,outputs_train = sample_data(FAST_sim_ind_train,sampling_type = 'KM',n_samples = 2)

    
    nc = len(reqd_controls)
    nout = len(reqd_outputs)
    controls_train = model_inputs[:,:nc]
    n_pts = len(controls_train);print(n_pts)
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

    
    # controls_train_sc = np.reshape(controls_train_sc,[n_exp,int(n_pts/n_exp),nc],order = 'F')
    # outputs_train_sc = np.reshape(outputs_train_sc,[n_exp,int(n_pts/n_exp),ny],order = 'F') 
    
    # controls_train_sc = controls_train_sc[:,:-1,:];print(controls_train_sc.shape)
    # outputs_train_sc = outputs_train_sc[:,:-1,:]
    #breakpoint()
    n_batch = 60001*n_exp*len(w)
    n_fac = n_batch/n_exp
    n_pts = controls_train_sc.shape[1]
    
    controls_train_sc = np.reshape(controls_train_sc,[n_batch,1,nc],order = 'F')
    outputs_train_sc = np.reshape(outputs_train_sc,[n_batch,1,ny],order = 'F')

    # controls_train_sc = np.reshape(controls_train_sc,[n_exp*len(w),60001,1,nc],order = 'F')
    # outputs_train_sc = np.reshape(outputs_train_sc,[n_exp*len(w),60001,1,ny],order = 'F')

    print(controls_train_sc.shape)
    print(outputs_train_sc.shape)


    model_name = 'FOWT_1p6_lstm_'+'R'+ '.keras'
    t1 = timer.time()
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        # 
        #model = create_model(16,n_batch,ny,controls_train_sc.shape)

        # model = Sequential()
        # model.add(LSTM(16))
        # model.add(Dense(ny))
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(ny))
        model.add(tf.keras.layers.Dense(ny))
        model.compile(optimizer = 'adam',loss = 'mse')
        model.fit(controls_train_sc,outputs_train_sc,batch_size = 1,epochs = 4,verbose = 2)

        model.save(model_name)

    t2 = timer.time()
    print(t2-t1)                    


    #breakpoint()
    #_,_,_,model_inputs,state_derivatives,outputs_test = sample_data(FAST_sim_ind[5],sampling_type = 'KM',n_samples = 2)
    ind_test_list  = [8]
    dt_list = [0.5]
    n_dt = len(dt_list)
    n_test = len(ind_test_list)

    bp_mse = np.zeros((n_dt,n_test))
    twbyt_mse = np.zeros((n_dt,n_test))
    gs_mse = np.zeros((n_dt,n_test))

    ctrl_type = 'CL'
    save_flag = True; plot_path = 'LSTM-results3'

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
                #dt_ = 1
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
                print(t2-t1)
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
                    ax.set_xlim([0,600])


                    if output == 'GenSpeed':
                            gs_of = outputs_test[:,i_out]
                            gs_dfsm = CubicSpline(time_test,outputs_test_pred[:,i_out])
                            gs_dfsm = gs_dfsm(time)

                            gs_mse[idt,i_test] = calculate_mse(gs_of,gs_dfsm)

                            results_dict['time'] = time
                            results_dict['genspeed_of'] = gs_of 
                            results_dict['genspeed_dfsm'] = gs_dfsm

                    if output == 'TwrBsMyt':
                            tbm_of = outputs_test[:,i_out]
                            tbm_dfsm = CubicSpline(time_test,outputs_test_pred[:,i_out])
                            tbm_dfsm = tbm_dfsm(time)

                            results_dict['twrbsmyt_of'] = gs_of 
                            results_dict['twrbsmyt_dfsm'] = gs_dfsm

                            twbyt_mse[idt,i_test] = calculate_mse(tbm_of,tbm_dfsm)


                    if save_flag:
                        if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                
                        fig.savefig(plot_path +os.sep+ output +'_' +str(i_test)+ '_comp.pdf')
                        plt.close(fig)

                if ctrl_type == 'CL':

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

                            bp_mse[idt,i_test] = calculate_mse(bp_of,bp_dfsm)



                        if save_flag:
                            if not os.path.exists(plot_path):
                                os.makedirs(plot_path)
                    
                            fig.savefig(plot_path +os.sep+ control +'_' +str(i_test)+ '_comp.pdf')
                            plt.close(fig)

    
    pkl_name = 'results_dict_8.pkl'

    with open(pkl_name,'wb') as handle:
        pickle.dump(results_dict,handle)

    plt.show()

    breakpoint()