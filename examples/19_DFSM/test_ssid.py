import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os
import time as timer
from mat4py import loadmat
from scipy.interpolate import CubicSpline
from weis.dfsm.dfsm_utilities import calculate_MSE as calculate_mse

# ROSCO toolbox modules 
from rosco.toolbox import control_interface as ROSCO_ci
from rosco import discon_lib_path
from rosco.toolbox.inputs.validation import load_rosco_yaml
from rosco.toolbox.utilities import write_DISCON
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import controller as ROSCO_controller

# DFSM modules
from weis.dfsm.ode_algorithms import RK4
from scipy.interpolate import interp1d

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension,reorganize_data
from weis.dfsm.construct_dfsm import DFSM
from weis.dfsm.dfsm_sample_data import sample_data
from rosco.toolbox.ofTools.util import spectral

if __name__ == '__main__':

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # path to DISCON library
    lib_name = discon_lib_path

    # Write parameter input file
    param_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_DISCON.IN')

    

    plot_flag = True;save_flag = True;

    # load SS model
    pkl_name = this_dir + os.sep +'n4sid_19_all'+os.sep+'SS_model.pkl'

    # load dfsm model
    with open(pkl_name,'rb') as handle:
        SS_model = pickle.load(handle)

    A = np.array(SS_model['A'])
    B = np.array(SS_model['B'])
    C = np.array(SS_model['C'])
    w = SS_model['w'];print(w)

    plot_path = 'n4sid_'+str(w)+ '_all2'
    outputs_max = np.array(SS_model['outputs_max'])

    reqd_controls = SS_model['reqd_controls']
    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    reqd_outputs = SS_model['reqd_outputs']

    testpath = '/home/athulsun/DFSM/data/FOWT_1p6'
    

    outfiles_test = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)

    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,1,100]),
                  'control_scaling_factor': np.array([1,1000,1,1]),
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

    ind_w = (w_unique == w)

    FAST_sim_ind = np.squeeze(FAST_sim[:,ind_w])

    time = FAST_sim_ind[0]['time']
    tspan = [time[0],time[-1]]
    nt = len(time)

    x0 = np.squeeze(SS_model['x0'])

    nx,nu = np.shape(B)
    ny = len(reqd_outputs)

    test_inds = [8] #[0,1,2,3,4,5,6,7,8,9]
    n_test = len(test_inds)
    bp_mse = np.zeros((n_test,))
    twbyt_mse = np.zeros((n_test,))
    gs_mse = np.zeros((n_test,))

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 18
    fontsize_tick = 12

    bp0 = 18
    gt0 = 19000 

    # parameters for ROSCO
    dt = time[1] - time[0]
    num_blade = int(3)
    args = {'DT': dt, 'num_blade': int(3)}

    genspeed_ind = reqd_outputs.index('GenSpeed')
    fa_acc_ind = reqd_outputs.index('NcIMURAys')
    gen_torque_ind = reqd_controls.index('GenTq')
    blade_pitch_ind = reqd_controls.index('BldPitch1')

    rpm2RadSec = 2.0*(np.pi)/60.0
    KWatt2Watt = 1000
    Deg2Rad = np.pi/180

    ctrl_type = 'CL'

    
    sim_time = []


    for i_test,ind in enumerate(test_inds):

        GS_of = []
        GS_ssid = []
        FA_of = []
        FA_ssid = []
        T = []

        test_case = FAST_sim_ind[ind]

        controls = test_case['controls']
        outputs = test_case['outputs']

        wind_speed = controls[:,0]
        wave_elev = controls[:,-1]

        wind_fun = CubicSpline(time,wind_speed)
        wave_fun = CubicSpline(time,wave_elev)

        X = np.zeros((nt+1,nx))
        Y = np.zeros((nt,ny))
        U = np.zeros((nt,nu))

        X[0,:] = x0

        # Load controller library
        if ctrl_type == 'CL':
            args = {'DT': dt, 'num_blade': int(3),'pitch':controls[0,blade_pitch_ind]}
            controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)

        t1 = timer.time()
        for i in range(nt):

            t = time[i]

            if t == 0 or ctrl_type == 'OL':

                U[i,:] = controls[i,:]

            else:
                turbine_state = {}

                if t == time[-1]:
                    turbine_state['iStatus'] = -1

                else:
                    turbine_state['iStatus'] = 1

                turbine_state['ws'] = controls[i-1,0] # estimate wind speed
                turbine_state['bld_pitch'] = np.deg2rad(U[i-1,blade_pitch_ind]) # blade pitch
                turbine_state['gen_torque'] = U[i-1,gen_torque_ind]*KWatt2Watt*1000 # generator torque

                turbine_state['num_blades'] = int(3) # number of blades
                turbine_state['t'] = t # current time step
                turbine_state['dt'] = dt # step size
                turbine_state['gen_eff'] = 95.89835000000/100 # generator efficiency
                
                turbine_state['gen_speed'] = Y[i-1,genspeed_ind]*rpm2RadSec*1 # generator speed
                turbine_state['rot_speed'] = Y[i-1,genspeed_ind]*rpm2RadSec*1# rotor speed
                turbine_state['NacIMU_FA_Acc'] = Y[i-1,fa_acc_ind]*np.deg2rad(1)

                turbine_state['Yaw_fromNorth'] = 0 # yaw
                turbine_state['Y_MeasErr'] = 0
                

                gen_torque, bld_pitch, nac_yawrate = controller_interface.call_controller(turbine_state)

                U[i,0] = wind_fun(t)
                U[i,gen_torque_ind] = gen_torque/(KWatt2Watt*1000)
                U[i,blade_pitch_ind] = np.rad2deg(bld_pitch)
                U[i,-1] = wave_fun(t)

            x = X[i,:]

            Xi = np.dot(A,x) + np.dot(B,U[i,:])

            Y[i,:] = np.dot(C,x)*outputs_max

            X[i+1,:] = Xi
        t2 = timer.time()

        sim_time.append(t2-t1)

        GS_ssid.append(Y[:,genspeed_ind])
        GS_of.append(outputs[:,genspeed_ind])
        FA_ssid.append(Y[:,fa_acc_ind])
        FA_of.append(outputs[:,fa_acc_ind])
        T.append(time)
        
        if ctrl_type == 'CL':
            controller_interface.kill_discon()

        if plot_flag:

            if ctrl_type == 'CL':

                for iu,control in enumerate(reqd_controls):

                    if control == 'BldPitch1':
                        
                        bp_of = controls[:,iu]
                        bp_dfsm = U[:,iu]
                       

                        bp_mse[i_test] = calculate_mse(bp_of,bp_dfsm)




                    fig,ax = plt.subplots(1)
                    ax.plot(time,controls[:,iu],label = 'OpenFAST')
                    ax.plot(time,U[:,iu],label = 'n4sid')
                    
                    
                    ax.set_title(control,fontsize = fontsize_axlabel)
                    ax.set_xlim(tspan)
                    ax.tick_params(labelsize=fontsize_tick)
                    ax.legend(ncol = 2,fontsize = fontsize_legend)
                    ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)

                    if save_flag:
                        if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                
                        fig.savefig(plot_path +os.sep+ control +'_' +str(i_test)+ '_comp.pdf')
                    plt.close(fig)

            ylim_list = [[5,10],[-1,1],[-2e5,4e5],[-1,6],[-0.8,0.8]]
            for iy,output in enumerate(reqd_outputs):

                if output == 'GenSpeed':
                    gs_of = outputs[:,iy]
                    gs_dfsm = Y[:,iy]

                    gs_mse[i_test] = calculate_mse(gs_of,gs_dfsm)

                if output == 'TwrBsMyt':
                    gs_of = outputs[:,iy]
                    gs_dfsm = Y[:,iy]

                    twbyt_mse[i_test] = calculate_mse(gs_of,gs_dfsm)

                fig,ax = plt.subplots(1)
                ax.plot(time,outputs[:,iy],label = 'OpenFAST')
                ax.plot(time,Y[:,iy],label = 'n4sid')
                
                
                ax.set_title(output,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.set_ylim(ylim_list[iy])
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)

                if save_flag:
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
            
                    fig.savefig(plot_path +os.sep+ output +'_' +str(i_test)+ '_comp.pdf')
                plt.close(fig)

        plot_psd = False

        if plot_psd:

            GS_of = np.squeeze(np.array(GS_of))
            GS_ssid = np.squeeze(np.array(GS_ssid))
            T = np.squeeze(np.array(T))

            fq_of,y_of,_ = spectral.fft_wrap(T,GS_of,averaging='Welch', averaging_window='Hamming', output_type='psd')
            fq_ssid,y_ssid,_ = spectral.fft_wrap(T,GS_ssid,averaging='Welch', averaging_window='Hamming', output_type='psd')

            fig,ax = plt.subplots(1)

            ax.plot(fq_of,y_of,label = 'OpenFAST')
            ax.plot(fq_ssid,y_ssid,label = 'N4sid')
            ax.legend(ncol = 2)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('Freq [Hz]')
            ax.set_ylabel('PSD GenSpeed')


            FA_of = np.squeeze(np.array(FA_of))
            FA_ssid = np.squeeze(np.array(FA_ssid))
           

            fq_of,y_of,_ = spectral.fft_wrap(T,FA_of,averaging='Welch', averaging_window='Hamming', output_type='psd')
            fq_ssid,y_ssid,_ = spectral.fft_wrap(T,FA_ssid,averaging='Welch', averaging_window='Hamming', output_type='psd')

            fig,ax = plt.subplots(1)

            ax.plot(fq_of,y_of,label = 'OpenFAST')
            ax.plot(fq_ssid,y_ssid,label = 'N4sid')
            ax.legend(ncol = 2)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('Freq [Hz]')
            ax.set_ylabel('PSD FA_Acc')
        
    print(sim_time)

    if not save_flag:
        plt.show()

    else:
        with open(plot_path + os.sep +'SS_model.pkl','wb') as handle:
            pickle.dump(SS_model,handle)


    breakpoint()