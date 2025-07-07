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


# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 15


plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# get path to this directory
this_dir = os.path.dirname(os.path.realpath(__file__))
outputs_dir = this_dir + os.sep + 'outputs'

def ModelData():
    
    model_data = {}

    model_data['reqd_states'] = ['PtfmSurge','PtfmPitch','TTDspFA','GenSpeed'];ns = len(model_data['reqd_states'])
    model_data['reqd_controls'] =  ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev'];nc = len(model_data['reqd_controls'])
    model_data['reqd_outputs'] = ['TwrBsFxt','TwrBsMyt','NcIMURAys','GenPwr','GenSpeed','PtfmPitch','TTDspFA','PtfmSurge'] 

    model_data['datapath'] = outputs_dir+os.sep +'fowt_test_1p62'


    scale_args = {'state_scaling_factor': np.array([1,1,1,1]),
                  'control_scaling_factor': np.array([1,1000,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1,1,1,1])
                  }
    
    model_data['scale_args'] = scale_args

    model_data['nx'] = 2

    # current speed to start the optimization
    model_data['w_test'] = 14

    # indices to be used for training and testing
    model_data['test_inds'] = np.arange(0,10)
    
    # training indices
    model_data['train_inds'] = np.arange(0,5)

    # pickle file to save the DFSM to
    model_data['n4sid_model_file'] = 'n4sid_model.pkl'

    # Use simulations from tmin seconds
    model_data['tmin'] = 00

    model_data['dt_extract'] = 0.01
     
    return model_data



if __name__ == '__main__':
    
    
    # datapath
    region = 'LPV'

    model_data = ModelData()
    datapath =  model_data['datapath'] #this_dir + os.sep + 'outputs' + os.sep + 'FOWT_1p6' #+ os.sep + 'openfast_runs/rank_0'
    param_filename = datapath + os.sep + 'DLC1.6_0_weis_job_00_DISCON.IN'
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
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=model_data['tmin']
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

    n_test = len(test_inds)
    bp_mse = np.zeros((n_test,))
    twrbsmyt_mse = np.zeros((n_test,))
    gs_mse = np.zeros((n_test,))

    model_sim_time = np.zeros((n_test,))

    n4sid_model_file = this_dir + os.sep + 'sys_id_test/nx_10/w_14'+os.sep + model_data['n4sid_model_file']

    with open(n4sid_model_file,'rb') as handle:
        n4sid_model = pickle.load(handle)

    A = n4sid_model['A']
    B = n4sid_model['B']
    C = n4sid_model['C']
    x0 = n4sid_model['x0']
    outputs_max = n4sid_model['outputs_max']

    nx = A.shape[0]

    ctrl_type = 'CL'

    gs_ind = reqd_outputs.index('GenSpeed')
    faacc_ind = reqd_outputs.index('NcIMURAys')
    blade_pitch_ind = reqd_controls.index('BldPitch1')
    gen_tq_ind = reqd_controls.index('GenTq')

    # path to DISCON library
    lib_name = discon_lib_path

    save_flag = True
    plot_flag = True 
    ts_path = 'sys_id_test' + os.sep + 'CL_validation'
    t_transition = 200


    for idx,ind in enumerate(test_inds):
                
        FS = FAST_sim[ind,ind_w]

        
        U_of = FS['controls']
        outputs_OF = FS['outputs']

        U_n4sid = np.zeros(U_of.shape)

        U_n4sid[0,:] = U_of[0,:]

        #time = results_mat['time']
        time = FAST_sim[ind,ind_w]['time']
        
        tf = time[-1]

        nt = len(time)

        # initalize
        X = np.zeros((nt+1,nx))
        Y = np.zeros((nt,ny))
        
        X[0,:] = x0[:,1]

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

        t1 = timer.time()
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
        model_sim_time[idx] = t2-t1
        print(t2-t1)

        if ctrl_type == 'CL':
            controller_interface.kill_discon()


            t_ind = (time >= t_transition)
            time = time[t_ind]; time = time - t_transition
            #time_dfsm = time_dfsm[t_ind]; time_dfsm = time_dfsm - t_transition

            Y = Y[t_ind,:];outputs_OF = outputs_OF[t_ind,:]
            U_n4sid = U_n4sid[t_ind,:];U_of = U_of[t_ind,:]
            

            gt_mse_w = calculate_MSE(U_of[:,blade_pitch_ind],U_n4sid[:,blade_pitch_ind])
            bp_mse_w = calculate_MSE(U_of[:,gen_tq_ind],U_n4sid[:,gen_tq_ind])
        
        if plot_flag:


            fig,ax = plt.subplots(1)
            ax.plot(time,U_of[:,0])
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            ax.set_title('RtVAvgxh',fontsize = fontsize_axlabel)
            ax.set_xlim([0,600])

            if save_flag:
                if not os.path.exists(ts_path):
                    os.makedirs(ts_path)
        
                fig.savefig(ts_path +os.sep+ 'RtVAvgxh_' +str(idx)+ '.pdf')
            plt.close(fig)

            if ctrl_type == 'CL':

                for iu in [1,2]:

                    if reqd_controls[iu] == 'BldPitch1':
                        bp_of = U_of[:,iu]
                        bp_n4sid = U_n4sid[:,iu]
                        
                        bp_mse[idx] = calculate_MSE(bp_of,bp_n4sid)

                        title = 'Blade Pitch [deg]'
                    else:
                        title = reqd_controls[iu]

                    fig,ax = plt.subplots(1)
                    ax.plot(time,U_of[:,iu],color = 'k',label = 'OpenFAST')
                    ax.plot(time,U_n4sid[:,iu],color = 'tab:orange',label = 'n4sid')
                    ax.tick_params(labelsize=fontsize_tick)
                    ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                    ax.set_title(title,fontsize = fontsize_axlabel)
                    ax.legend(ncol = 2,fontsize = fontsize_legend)
                    ax.set_xlim([0,600])

                    if save_flag:
                        if not os.path.exists(ts_path):
                            os.makedirs(ts_path)
                
                        fig.savefig(ts_path +os.sep+ reqd_controls[iu]+'_' +str(idx)+ '.pdf')
                    plt.close(fig)

            for i in range(ny):

                if reqd_outputs[i] == 'GenSpeed':
                    gs_of = outputs_OF[:,i]
                    gs_n4sid = Y[:,i]
                    
                    gs_mse[idx] = calculate_MSE(gs_of,gs_n4sid)

                    title = 'Generator Speed [rpm]'
                else:
                    title = reqd_outputs[i]

                if reqd_outputs[i] == 'TwrBsMyt':
                    M_of = outputs_OF[:,i]
                    M_n4sid = Y[:,i]
                    
                    twrbsmyt_mse[idx] = calculate_MSE(M_of,M_n4sid)

                    title = 'Tower-Base Moment [kNm]'
                else:
                    title = reqd_outputs[i]

                fig,ax1 = plt.subplots(1)
                
                ax1.plot(time,outputs_OF[:,i],color = 'k',label = 'OpenFAST')
                ax1.plot(time,Y[:,i],color = 'tab:orange',label = 'n4sid')
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                ax1.set_xlim([0,600])
                
                ax1.tick_params(labelsize=fontsize_tick)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                ax1.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                ax1.set_title(reqd_outputs[i],fontsize = fontsize_axlabel)

                if save_flag:
                    if not os.path.exists(ts_path):
                        os.makedirs(ts_path)
            
                    fig.savefig(ts_path +os.sep+ reqd_outputs[i] +'_' +str(idx)+ '_comp.pdf')
                plt.close(fig)

    results_dict = {'bp_mse':bp_mse,'gs_mse':gs_mse,'twrbsmyt_mse':twrbsmyt_mse,'model_sim_time':model_sim_time}
    results_file = ts_path+os.sep+'results_dict.pkl'

    
    with open(results_file,'wb') as handle:
        pickle.dump(results_dict,handle)

    plt.show()


    