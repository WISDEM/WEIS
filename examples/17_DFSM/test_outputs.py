import matplotlib.pyplot as plt 
import numpy as np
import os, platform
import sys

# ROSCO toolbox modules 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import control_interface as ROSCO_ci
from ROSCO_toolbox.utilities import write_DISCON
from ROSCO_toolbox.inputs.validation import load_rosco_yaml
from scipy.interpolate import CubicSpline,interp1d
from scipy.interpolate import CubicSpline
from ROSCO_toolbox.ofTools.util import spectral
import time as timer

# DFSM modules
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension,calculate_time, calculate_MSE
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
from dfsm.evaluate_dfsm import evaluate_dfsm
from scipy.integrate import solve_ivp

from dfsm.dfsm_rosco_simulation import run_sim_ROSCO


def extrapolate_controls(t_out,u, t):

    # extract time
    t1 = t[0]

    # scale time
    t -= t1 
    t_out = t_out - t1

    if len(t) == 2:
        # evaluate slope
        u1 = u[0]; u2 = u[1]
        b0 = (u2 - u1)/t[1]

        # extrapolate
        u_out = u1 + b0*t_out

    elif len(t) == 3:
         
        u1 = u[0];u2 = u[1]; u3 = u[2]

        b0 = (t[2]**2*(u1 - u2) + t[1]**2*(-u1 + u3))/(t[1]*t[2]*(t[1] - t[2]))
        c0 = ( (t[1]-t[2])*u1+ t[2]*u2 - t[1]*u3 ) / (t[1]*t[2]*(t[1] - t[2]))

        u_out = u1 + b0*t_out + c0*t_out**2
         

    return u_out


def RK4(x0, dt, tspan, DFSM, param):

    # calculate intervals
    t0 = tspan[0]; tf = tspan[1]

    # initialize
    T = np.arange(t0,tf+dt,dt)
    n = len(T); nx = len(x0)
    X = np.zeros((n,nx))

    ny = param['ny']

    if ny == 0:
        Y = []
    else:
        Y = np.zeros((n,ny))
         
   
    # starting point
    X[0,:] = x0 
    
    # parameter to scale generator speed
    gen_speed_scaling = param['gen_speed_scaling']

    # extract current speed and wave function
    wave_fun = param['wave_fun']
    wind_fun = param['w_fun']

    # evaluate wind/current speed for the time stencil
    WS = wind_fun(T)

    # convert rpm to rad/s
    # OpenFAST stores genspeed as rpm, whearas ROSCO requiers genspeed in rad/s
    rpm2RadSec = 2.0*(np.pi)/60.0


    # initialize extrapolation arrays
    if wave_fun == None:
        U_extrap = np.zeros((2*n-1,3))
        U = np.zeros((n,3))

    else:
        WE = wave_fun(T)
        U_extrap = np.zeros((2*n-1,4))
        U = np.zeros((n,4))

    # storage array
    ind_extrap = 0
    T_extrap = np.zeros((2*n-1,))
    T_extrap[ind_extrap] = T[0]

    # set first entry in the extrapolation 
    U_extrap[ind_extrap,0] = WS[0]
    U_extrap[ind_extrap,1] = param['gen_torque'][0]
    U_extrap[ind_extrap,2] = param['blade_pitch'][0]
    
    # store control
    U[0,0] = WS[0]
    U[0,1] = param['gen_torque'][0]
    U[0,2] = param['blade_pitch'][0]

    UK4 = []

    if not(wave_fun == None):
        U[0,3] = WE[0]
        U_extrap[ind_extrap,3] = WE[0]
   
    # loop through and evaluate states
    for h in range(1,n):

        t_n = T[h-1]
        x_n = X[h-1,:]

        # get from previous time step
        u_k1 = U[h-1,:]

        # inputs for DFSM
        inputs = np.hstack([u_k1,x_n])

        k1 = evaluate_dfsm(DFSM,inputs,'deriv')

        if h == 1:
            u_k2 = u_k1 # zero order hold for the first step
        else:
            #breakpoint()
            u2_ =  u_k1
            u1_ = U_extrap[ind_extrap-1,:]
            t2_ = T_extrap[ind_extrap]
            t1_ = T_extrap[ind_extrap-1]
            u_k2 = extrapolate_controls(t_n + dt/2,[u1_,u2_],[t1_,t2_])

        # add values to exrap array
        ind_extrap += 1
        U_extrap[ind_extrap] = u_k2
        T_extrap[ind_extrap] = t_n + dt/2


        xn_k2 = x_n + dt/2*k1
        inputs = np.hstack([u_k2,xn_k2])
        k2 = evaluate_dfsm(DFSM,inputs,'deriv')

        u_k3 = u_k2
        xn_k3 =  x_n + dt/2*k2
        inputs = np.hstack([u_k3,xn_k3])
        k3 = evaluate_dfsm(DFSM,inputs,'deriv')

        
        u2_ = U_extrap[ind_extrap,:]
        u1_ = U_extrap[ind_extrap-1,:]
        t2_ = T_extrap[ind_extrap]
        t1_ = T_extrap[ind_extrap-1]
        u_k4 = extrapolate_controls(t_n + dt,[u1_, u2_],[t1_, t2_])
        
        xn_k4 = x_n + dt*k3
        inputs = np.hstack([u_k4,xn_k4])
        k4 = evaluate_dfsm(DFSM,inputs,'deriv')

        x_step = x_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        #UK4.append(u_k4)
        inputs = np.hstack([u_k4,x_step])

        if ny > 0:
            Y[h,:] = evaluate_dfsm(DFSM,inputs,'outputs')
        
        X[h,:] = x_step

        turbine_state = {}

        if h == n:
            turbine_state['iStatus'] = -1
        else:
            turbine_state['iStatus'] = 1

        turbine_state['bld_pitch'] = np.deg2rad(U[h-1,2])
        turbine_state['gen_torque'] = U[h-1,1]*1000
        turbine_state['t'] = t_n
        turbine_state['dt'] = dt 
        turbine_state['ws'] = WS[h]
        turbine_state['num_blades'] = int(2)
        turbine_state['gen_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling
        turbine_state['gen_eff'] = param['VS_GenEff']/100
        turbine_state['rot_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling/param['WE_GearboxRatio']
        turbine_state['Yaw_fromNorth'] = 0
        turbine_state['Y_MeasErr'] = 0

        if not(DFSM.FA_Acc_ind_s == None):
            turbine_state['FA_Acc'] = X[h,DFSM.FA_Acc_ind_s]/1

        elif not(DFSM.FA_Acc_ind_o == None):
            turbine_state['FA_Acc'] = Y[h,DFSM.FA_Acc_ind_o]/1
        
        if not(DFSM.NacIMU_FA_Acc_ind_s == None):
            turbine_state['NacIMU_FA_Acc'] = X[h,DFSM.NacIMU_FA_Acc_ind_s]

        elif not(DFSM.NacIMU_FA_Acc_ind_o == None):
            turbine_state['NacIMU_FA_Acc'] = Y[h,DFSM.NacIMU_FA_Acc_ind_o]

        # call ROSCO to get control values
        gen_torque, bld_pitch, nac_yawrate = param['controller_interface'].call_controller(turbine_state)

        # convert to right units
        gen_torque = gen_torque/1000
        #gen_torque = param['gen_torque'][0]
        bld_pitch = np.rad2deg(bld_pitch)

        U[h,0] = WS[h]
        U[h,1] = gen_torque 
        U[h,2] = bld_pitch

        if not(wave_fun == None):
            U[h,3] = WE[h]

        ind_extrap += 1
        U_extrap[ind_extrap,:] = U[h,:]
        T_extrap[ind_extrap] = t_n + dt

    return T,X,U,Y, T_extrap, U_extrap

    

if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # parameter file
    parameter_filename = os.path.join(this_dir,'OpenFAST','RM1_MHK.yaml')
    
    # tune directory
    
    # load
    inps = load_rosco_yaml(parameter_filename)
    path_params         = inps['path_params']
    turbine_params      = inps['turbine_params']
    controller_params   = inps['controller_params']
    
    # path to DISCON library
    lib_name = os.path.join('/home/athulsun/WEIS-AKS/local/lib','libdiscon.so')
    
    # Load turbine data from openfast model
    turbine = ROSCO_turbine.Turbine(turbine_params)
    turbine.v_rated = 1.6789904628983487
    
    # cp ct cq file
    cp_filename = os.path.join(this_dir,'OpenFAST','MHK_RM1_Cp_Ct_Cq.txt') #os.path.join(this_dir,path_params['FAST_directory'],path_params['rotor_performance_filename'])
    
    # load turbine    
    turbine.load_from_fast(
        path_params['FAST_InputFile'],
        os.path.join(this_dir,path_params['FAST_directory']),
        rot_source='txt',txt_filename=cp_filename
        )
    
    # Tune controller 
    controller      = ROSCO_controller.Controller(controller_params)
    controller.tune_controller(turbine)
    
    # Write parameter input file
    param_filename = os.path.join(this_dir,'DISCON.IN')
    write_DISCON(
      turbine,controller,
      param_file=param_filename, 
      txt_filename=cp_filename
      )
    
    # datapath
    region = 'R'
    datapath = this_dir + os.sep + 'outputs' + os.sep + 'MHK_BR_10' + os.sep + 'openfast_runs/rank_0'
    
    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)
    
    # required states
    reqd_states = ['PtfmPitch','GenSpeed']
    
    state_props = {'units' : ['[deg]','[rpm]','[m/s2]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = ['YawBrTAxp','GenPwr','NcIMURAys'] #,'TwrBsMyt'
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1,1])
                  }
    
    # filter parameters
    filter_args = {'states_filter_flag': [False,False,False],
                   'states_filter_type': [[],[],[]],
                   'states_filter_tf': [[],[],[]],
                   'controls_filter_flag': [False,False,False],
                   'controls_filter_tf': [0,0,0],
                   'outputs_filter_flag': []
                   }
    
    # name of mat file that has the linearized models
    mat_file_name = None #this_dir + os.sep + 'linear-models.mat'
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    
    
    save_path = 'plots_ROSCO'
    # load and process data
    sim_detail.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail.FAST_sim
    
    # plot data
    #plot_inputs(sim_detail,4,'separate')
    
    # split of training-testing data
    n_samples = 20
    test_inds = [0]
    
    # construct surrogate model
    dfsm_model = DFSM(sim_detail,n_samples = n_samples,L_type = 'LTI',N_type = None, train_split = 0.4)
    dfsm_model.construct_surrogate()
    dfsm_model.simulation_time = []

    GT_OF_all = []
    BP_OF_all = []
    BP_DFSM_all = []
    GT_DFSM_all = []
        
    for idx,ind in enumerate(test_inds):
            
        test_data = dfsm_model.test_data[ind]
        bp_of = test_data['controls'][:,2]
        gt_of = test_data['controls'][:,1]
        

        nt = 1000
        wind_speed = test_data['controls'][:,0] #
        nt = len(wind_speed)
        t0 = 0;tf = 700
        dt = 0.0; t1 = t0 + dt
        time_of = np.linspace(t0,tf,nt)
        
        w_pp = CubicSpline(time_of, wind_speed)
        w_fun = lambda t: w_pp(t)
        w0 = w_fun(0)
        bp0 = 9
        gt0 = 0 #8.3033588
        
        if reqd_controls[-1] == 'Wave1Elev':
            wave_elev = test_data['controls'][:,3]
            wave_pp = CubicSpline(time_of,wave_elev)
            wave_fun = lambda t:wave_pp(t)

        else:
            wave_fun = None
        #breakpoint()
        tspan = [t1,tf]
        
        x0 = test_data['states'][0,:]
        x0_m = np.mean(test_data['states'],0)
        states_ = test_data['states']
        if len(reqd_outputs)>0:
            outputs_of = test_data['outputs']
        
        #x0[2] = 0

        dt = 0.001
        num_blade = int(2)

        args = {'DT': dt, 'num_blade': int(2)}


        # Load controller library
        controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)
        
        if True: 
            # hardcoded for now
            param = {'VS_GenEff':94.40000000000,
                     'WE_GearboxRatio':53.0,
                     'VS_RtPwr':500,
                     'time':[t0],
                     'dt':[dt],
                     'blade_pitch':[bp0],
                     'gen_torque':[gt0],
                     't0':t1,
                     'tf':tf,
                     'w_fun':w_fun,
                     'gen_speed_scaling':1,
                     'controller_interface':controller_interface,
                     'wave_fun':wave_fun,
                     'ny': len(reqd_outputs)
                     }
            
            # solver method and options
            solve_options = {'method':'RK45','rtol':1e-6,'atol':1e-6}
            
            # start timer and solve for the states and controls
            t1 = timer.time()
            T_dfsm, X_dfsm, U_dfsm,Y_dfsm,T_extrap, U_extrap = RK4(x0, dt, tspan, dfsm_model, param)
            t2 = timer.time()
            dfsm_model.simulation_time.append(t2-t1)
            
            # extract solution
            time = T_dfsm

            states = X_dfsm
            #breakpoint()
            states = interp1d(time,states,axis = 0)(time_of)
            #states = states.T
            
            # kill controller
            param['controller_interface'].kill_discon()
            
            tspan = [0,tf]
            time_sim = T_dfsm
            blade_pitch = U_dfsm[:,2]
            gen_torque = U_dfsm[:,1]
            
            # interpolate controls
            blade_pitch = interp1d(time_sim,blade_pitch,axis = 0)(time_of)
            gen_torque = interp1d(time_sim,gen_torque,axis = 0)(time_of)
            time = time_of
            # interpolate time
            wind_of = w_fun(time_of)
            blade_pitch_of = bp_of
            gen_torque_of = gt_of
            
            #wave_elev = wave_fun(time)
            
            controls_of = np.array([wind_of,gen_torque_of,blade_pitch_of]).T
            controls_dfsm = np.array([wind_of,gen_torque,blade_pitch]).T
            
            inputs_dfsm = np.hstack([controls_dfsm,states])
            states_of = CubicSpline(time_of,states_)(time)
            
            X_list = [{'time':time,'names':test_data['state_names'],'n':test_data['n_states'],'OpenFAST':states_of,
                      'DFSM':states,'units':['[rpm]','[rpm/s]'],'key_freq_name':[['2P']],'key_freq_val':[[0.39]]}]
            
            U_list = [{'time': time, 'names': test_data['control_names'],'n': test_data['n_controls']
                      ,'OpenFAST':controls_of,'DFSM':controls_dfsm,'units': ['[m/s]','[kNm]','[deg]']}]
            
            fun_type = 'outputs'
            
                
            #----------------------------------------------------------------------
            save_flag = False;plot_path = 'plots_ROSCO_TR_nw_new'
            BP_OF_all.append(bp_of)
            BP_DFSM_all.append(blade_pitch)

            # plot properties
            markersize = 10
            linewidth = 1.5
            fontsize_legend = 16
            fontsize_axlabel = 18
            fontsize_tick = 12

            #------------------------------------------------
            # Plot Blade Pitch
            #-----------------------------------------------
            
            fig,ax = plt.subplots(1)
            ax.plot(time,blade_pitch,label = 'DFSM')
            ax.plot(time_of,bp_of,label = 'OpenFAST')
            ax.set_title('BldPitch [deg]',fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            #ax.set_ylim([-1,5])
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'blade_pitch' +str(idx) +'_comp.pdf')

            #------------------------------------------------------
            # Plot GenSpeed
            #------------------------------------------------------
            
            fig,ax = plt.subplots(1)
            ax.plot(time,states[:,1],label = 'DFSM')
            ax.plot(time_of,states_[:,1],label = 'OpenFAST')
            ax.set_title('GenSpeed [rpm]',fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            #ax.set_ylim([580,630])
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'gen_speed' +str(idx) + '_comp.pdf')

            #------------------------------------------------------
            # Plot PtfmPitch
            #------------------------------------------------------
            if reqd_states[0] == 'PtfmPitch':
                fig,ax = plt.subplots(1)
                ax.plot(time,states[:,0],label = 'DFSM')
                ax.plot(time_of,states_[:,0],label = 'OpenFAST')
                ax.set_title('PtfmPitch [deg]',fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                #ax.set_ylim([580,630])
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ 'ptfm_pitch' +str(idx) + '_comp.pdf')

            #------------------------------------------------------
            # Plot YawBrTAxp
            #------------------------------------------------------
            if reqd_states[-1] == 'YawBrTAxp':

                fig,ax = plt.subplots(1)
                ax.plot(time,states[:,2],label = 'DFSM')
                ax.plot(time_of,states_[:,2],label = 'OpenFAST')
                ax.set_title('YawBrTAxp [m/s2]',fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                #ax.set_ylim([580,630])
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ 'yawbrtaxp' +str(idx) + '_comp.pdf')
                
            #------------------------------------------
            # Plot YawBrTAxp
            #--------------------------------
            
            if dfsm_model.n_outputs > 0:
                fig,ax = plt.subplots(1)
                ax.plot(T_dfsm,Y_dfsm[:,1],label = 'DFSM')
                ax.plot(time,outputs_of[:,1],label = 'OpenFAST')
                
                ax.set_title('GenPwr [kW]',fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                #ax.set_ylim([480,530])
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ 'GenPwr' +str(idx) + '_comp.pdf')


                fig,ax = plt.subplots(1)
                ax.plot(T_dfsm,Y_dfsm[:,0],label = 'DFSM')
                ax.plot(time,outputs_of[:,0],label = 'OpenFAST')
                
                ax.set_title('YawBrTAxp',fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                #ax.set_ylim([480,530])
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ 'yaw_br_taxp' +str(idx) + '_comp.pdf')

                fig,ax = plt.subplots(1)
                ax.plot(T_dfsm,Y_dfsm[:,2],label = 'DFSM')
                ax.plot(time,outputs_of[:,2],label = 'OpenFAST')
                
                ax.set_title('NcIMUys',fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                #ax.set_ylim([480,530])
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                # if save_flag:
                #     if not os.path.exists(plot_path):
                #             os.makedirs(plot_path)
                        
                #     fig.savefig(plot_path +os.sep+ 'nac_imu' +str(idx) + '_comp.pdf')
                
            GT_OF_all.append(gt_of)
            GT_DFSM_all.append(gen_torque)    
                
            fig,ax = plt.subplots(1)
            ax.plot(time,gen_torque,label = 'DFSM')
            ax.plot(time_of,gt_of,label = 'OpenFAST',alpha = 0.9)
            
            ax.set_title('GenTq [kNm]',fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            #ax.set_ylim([2,6])
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'GenTq' +str(idx) + '_comp.pdf')
            
            fig,ax = plt.subplots(1)
            ax.plot(time_of,w_fun(time_of))
            #ax.plot(time,gen_torque,label = 'DFSM')
            ax.set_title('Current Speed [m/s]',fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'FlowSpeed' +str(idx) + '_comp.pdf')

            # fig,ax = plt.subplots(1)

            # ax.plot(time,w_fun(time_of),label = 'ROSCO')
            # ax.plot(T_extrap,U_extrap[:,0],'.',alpha = 0.3,label = 'extrap')

            # ax.set_xlim(tspan)
            # ax.tick_params(labelsize=fontsize_tick)
            # ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            # ax.legend(ncol = 2)
            
            
            # fig,ax = plt.subplots(1)

            # ax.plot(time,gen_torque,label = 'ROSCO')
            # ax.plot(T_extrap,U_extrap[:,1],'.',alpha = 0.3,label = 'extrap')

            # ax.set_xlim(tspan)
            # ax.tick_params(labelsize=fontsize_tick)
            # ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            # ax.legend(ncol = 2)

            # fig,ax = plt.subplots(1)

            # ax.plot(time,blade_pitch,label = 'ROSCO')
            # ax.plot(T_extrap,U_extrap[:,2],'.',alpha = 0.3,label = 'extrap')

            # ax.set_xlim(tspan)
            # ax.tick_params(labelsize=fontsize_tick)
            # ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            # ax.legend(ncol = 2)


            # UK4 = np.array(UK4)

            # fig,ax = plt.subplots(1)

            
            # ax.plot(time_sim[:-1],UK4[:,2],'.',label = 'extrap')
            # ax.plot(T_dfsm,U_dfsm[:,2],'.',alpha = 0.3,label = 'act')

            # ax.set_xlim(tspan)
            # ax.tick_params(labelsize=fontsize_tick)
            # ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            # ax.legend(ncol = 2)


            # fig,ax = plt.subplots(1)

            
            # ax.plot(time_sim[:-1],UK4[:,1],'.',label = 'extrap')
            # ax.plot(T_dfsm,U_dfsm[:,1],'.',alpha = 0.3,label = 'act')

            # ax.set_xlim(tspan)
            # ax.tick_params(labelsize=fontsize_tick)
            # ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            # ax.legend(ncol = 2)

            # breakpoint()


            

    print('')
    print(dfsm_model.simulation_time)
    print('')


    
    #breakpoint()
    plt.show()
    
        
        
        
        
        


