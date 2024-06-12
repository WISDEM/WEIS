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
    
    #

     # datapath
    region = 'TR'
    datapath = this_dir + os.sep + 'outputs' + os.sep + 'simple_sim_R' + os.sep + 'openfast_runs/rank_0'
    
    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)
    
    # required states
    reqd_states = ['GenSpeed']
    
    state_props = {'units' : ['[deg]','[rpm]','[m/s2]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = ['TwrBsMyt','GenPwr','TwrBsMxt'] #['YawBrTAxp', 'NcIMURAys', 'GenPwr'] #,'TwrBsMyt'
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([100]),
                  'control_scaling_factor': np.array([1,1,1]),
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
    mat_file_name = None #this_dir + os.sep + 'DFSM_MHK_TR_nowave.mat'
    
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
        bp0 = 15
        gt0 = 8.3033588
        
        wave_elev = np.ones((nt,))*0 #test_data['controls'][:,3]
        wave_pp = CubicSpline(time_of,wave_elev)
        wave_fun = lambda t:wave_pp(t)
        
        tspan = [t1,tf]
        
        x0 = test_data['states'][0,:]
        x0_m = np.mean(test_data['states'],0)
        states_ = test_data['states']
        
        #x0[2] = 0
        # Load controller library
        #controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test')
        
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
                     'gen_speed_scaling':100,
                     'controller_interface':ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename),
                     'wave_fun':None
                     }
            
            # solver method and options
            solve_options = {'method':'RK45','rtol':1e-6,'atol':1e-6}
            
            # start timer and solve for the states and controls
            t1 = timer.time()
            sol =  solve_ivp(run_sim_ROSCO,tspan,x0,method=solve_options['method'],args = (dfsm_model,param),rtol = solve_options['rtol'],
                             atol = solve_options['atol'], t_eval = time_of)
            t2 = timer.time()
            dfsm_model.simulation_time.append(t2-t1)
            
            # extract solution
            time = sol.t

            states = sol.y
            states = states.T
            
            # kill controller
            param['controller_interface'].kill_discon()
            
            tspan = [0,tf]
            time_sim = param['time']
            blade_pitch = np.array(param['blade_pitch'])
            gen_torque = np.array(param['gen_torque'])
            
            # interpolate controls
            blade_pitch = interp1d(time_sim,blade_pitch)(time)
            gen_torque = interp1d(time_sim,gen_torque)(time)
            
            # interpolate time
            wind_of = w_fun(time)
            blade_pitch_of = interp1d(time_of,bp_of)(time)
            gen_torque_of = interp1d(time_of,gt_of)(time)
            
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
            
            if dfsm_model.n_outputs > 0:
                outputs_of = test_data['outputs']
                
                outputs_of = CubicSpline(time_of,outputs_of)(time)
                
                
                outputs_dfsm = evaluate_dfsm(dfsm_model,inputs_dfsm,fun_type)
                    
                Y_list = [{'time':time,'names':test_data['output_names'],'n':test_data['n_outputs'],
                          'OpenFAST':outputs_of,'DFSM':outputs_dfsm,'units':['[kW]','[kN]'],
                          'key_freq_name':[['2P'],['2P']],'key_freq_val':[[0.39],[0.39]]}]
                
            #----------------------------------------------------------------------
            save_flag = False;plot_path = 'plots_ROSCO_TR_nw'
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
                ax.plot(time,outputs_dfsm[:,1],label = 'DFSM')
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
                ax.plot(time,outputs_dfsm[:,0],label = 'DFSM')
                ax.plot(time,outputs_of[:,0],label = 'OpenFAST')
                
                ax.set_title('TwrBsMyt [kNm]',fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                #ax.set_ylim([480,530])
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ 'twr_bs_myt' +str(idx) + '_comp.pdf')
                
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
            

        
        # if save_flag:
        #     if not os.path.exists(plot_path):
        #             os.makedirs(plot_path)
                
        #     fig.savefig(plot_path +os.sep+ 'GenSpeed' + '_comp.svg')

    print('')
    print(dfsm_model.simulation_time)
    print('')

    # BP_OF_all = np.array(BP_OF_all); BP_OF_all = np.squeeze(BP_OF_all.reshape([-1,1]))
    # BP_DFSM_all = np.array(BP_DFSM_all); BP_DFSM_all = np.squeeze(BP_DFSM_all.reshape([-1,1]))
    # BP_all = (np.array([BP_DFSM_all,BP_OF_all])).T
    # GT_OF_all = np.array(GT_OF_all); GT_OF_all = np.squeeze(GT_OF_all.reshape([-1,1]))
    # GT_DFSM_all = np.array(GT_DFSM_all); GT_DFSM_all = np.squeeze(GT_DFSM_all.reshape([-1,1]))
    # GT_all = (np.array([GT_DFSM_all,GT_OF_all])).T

    # GT_MSE = calculate_MSE(GT_OF_all,GT_DFSM_all)
    # BP_MSE = calculate_MSE(BP_OF_all, BP_DFSM_all)

    
    #breakpoint()
    plt.show()
        
        
        
        
        
        


