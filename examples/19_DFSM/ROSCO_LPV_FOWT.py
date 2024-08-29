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
from scipy.interpolate import interp1d

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension
from weis.dfsm.construct_dfsm import DFSM

    

if __name__ == '__main__':

    diy_flag = True
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # path to DISCON library
    lib_name = discon_lib_path

    # Write parameter input file
    param_filename = '/home/athulsun/WEIS-AKS-commit/examples/19_DFSM/IEA_15_MW/IEA_w_TMD_DISCON.IN'
    
    plot_path = 'plot_dfsm_result';save_flag = False

    # pickle with the saved DFSM model
    pkl_name = this_dir + os.sep +'dfsm_1p6.pkl'

    # load dfsm model
    with open(pkl_name,'rb') as handle:
        dfsm = pickle.load(handle)
   
    # datapath
    region = 'LPV'

    testpath = '/home/athulsun/DFSM/data/1p6_test'
    

    outfiles_test = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)
    
    # required states
    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    
    state_props = {'units' : ['[deg]','[m/s2]','[rpm]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr'] #,'TwrBsMyt'
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,1,100]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1])
                  }
    
    # filter parameters
    filter_args = {'state_filter_flag': [True,True,True],
                   'state_filter_type': [['filtfilt'],['filtfilt'],['filtfilt']],
                   'state_filter_tf': [[0.1],[0.1],[0.1]],
                   'control_filter_flag': [False,False,False],
                   'control_filter_tf': [0,0,0],
                   'output_filter_flag': []
                   }
    
    
    sim_detail_test = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = [],region = region)
    
    save_path = 'plots_ROSCO'
    # load and process data
    
    sim_detail_test.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail_test.FAST_sim
    
    # plot data
    #plot_inputs(sim_detail,4,'separate')
    
    # split of training-testing data
    n_samples = 1
    test_inds = [9]


    if diy_flag:
         
        # tuning_yaml2 = os.path.join(this_dir,'../01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi/IEA15MW-UMaineSemi.yaml')
        # inputs2 = load_rosco_yaml(tuning_yaml2)

        # path_params2         = inputs2['path_params']
        # turbine_params2      = inputs2['turbine_params']
        # controller_params2   = inputs2['controller_params']
        # cp_filename = os.path.join(this_dir,'../01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi',path_params2['rotor_performance_filename'])


        # turbine2 = ROSCO_turbine.Turbine(turbine_params2)
        # turbine2.v_rated = 10.64
        # turbine2.load_from_fast(
        #     path_params2['FAST_InputFile'],
        #     os.path.join(this_dir,'../01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi'),
        #     rot_source='txt',txt_filename=cp_filename
        #     )
        
        # controller2      = ROSCO_controller.Controller(controller_params2)
        # controller2.tune_controller(turbine2)

        param_filename2 = os.path.join(this_dir,'outputs','FOWT_DFSM_test','IEA_w_TMD_0_DISCON.IN')
        # write_DISCON(
        # turbine2,controller2,
        # param_file=param_filename2, 
        # txt_filename=cp_filename
        # )
            
    
        
    for idx,ind in enumerate(test_inds):
            
        test_data = FAST_sim[ind]
        states_of = test_data['states']
        controls_of = test_data['controls']
        bp_of = test_data['controls'][:,2]
        gt_of = test_data['controls'][:,1]
        
        nt = 1000
        wind_speed = test_data['controls'][:,0] #
        nt = len(wind_speed)
        t0 = 0;tf = 600
        dt = 0.0; t1 = t0 + dt
        time_of = np.linspace(t0,tf,nt)
        
        w_pp = CubicSpline(time_of, wind_speed)
        w_fun = lambda t: w_pp(t)
        w0 = w_fun(0)
        bp0 = 7.8
        gt0 = 19000 #8.3033588
        
        if reqd_controls[-1] == 'Wave1Elev':
            wave_elev = test_data['controls'][:,3]
            wave_pp = CubicSpline(time_of,wave_elev)
            wave_fun = lambda t:wave_pp(t)

        else:
            wave_fun = None
        
        # time span
        tspan = [t1,tf]
        
        x0 = test_data['states'][0,:]
        x0_m = np.mean(test_data['states'],0)
        states_ = test_data['states']
        if len(reqd_outputs)>0:
            outputs_of = test_data['outputs']
        
        #x0[2] = 0

        dt = 0.01
        num_blade = int(3)

        args = {'DT': dt, 'num_blade': int(3),'pitch':bp0}


        # Load controller library
        controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)
        
        if True: 
            # hardcoded for now
            param = {'VS_GenEff':95.75600000000,
                     'WE_GearboxRatio':1.0,
                     'VS_RtPwr':15000000.00000,
                     'time':[t0],
                     'dt':[dt],
                     'blade_pitch':[bp0],
                     'gen_torque':[gt0],
                     't0':t1,
                     'tf':tf,
                     'w_fun':w_fun,
                     'gen_speed_scaling':scale_args['state_scaling_factor'][-1],
                     'controller_interface':controller_interface,
                     'wave_fun':wave_fun,
                     'ny': len(reqd_outputs)
                     }
            
            
        # start timer and solve for the states and controls
        t1 = timer.time()
        time_dfsm, states_dfsm, controls_dfsm,outputs_dfsm,T_extrap, U_extrap = RK4(x0, dt, tspan, dfsm, param)
        t2 = timer.time()
        dfsm.simulation_time.append(t2-t1)
        
        # shutdown controller
        param['controller_interface'].kill_discon()

        ncol = 2 
        if diy_flag:

            ncol = 1

            # Load controller library
            controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename2,sim_name='sim_test',**args)

            # hardcoded for now
            param = {'VS_GenEff':95.75600000000,
                     'WE_GearboxRatio':1.0,
                     'VS_RtPwr':15000000.00000,
                     'time':[t0],
                     'dt':[dt],
                     'blade_pitch':[bp0],
                     'gen_torque':[gt0],
                     't0':t1,
                     'tf':tf,
                     'w_fun':w_fun,
                     'gen_speed_scaling':scale_args['state_scaling_factor'][-1],
                     'controller_interface':controller_interface,
                     'wave_fun':wave_fun,
                     'ny': len(reqd_outputs)
                     }
            
            # start timer and solve for the states and controls
            t1 = timer.time()
            time_dfsm2, states_dfsm2, controls_dfsm2,outputs_dfsm2,T_extrap, U_extrap = RK4(x0, dt, tspan, dfsm, param)
            t2 = timer.time()
            dfsm.simulation_time.append(t2-t1)
            
            # shutdown controller
            param['controller_interface'].kill_discon()
             
    
    #----------------------------------------------------------------------
    # Plot results
    #----------------------------------------------------------------------

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 18
    fontsize_tick = 12

    #------------------------------------------------
    # Plot controls
    #-----------------------------------------------
    for iu,control in enumerate(reqd_controls):
            
        fig,ax = plt.subplots(1)
        ax.plot(time_of,controls_of[:,iu],label = 'OpenFAST')
        ax.plot(time_dfsm,controls_dfsm[:,iu],label = 'DFSM')

        if diy_flag:
            ax.plot(time_dfsm2,controls_dfsm2[:,iu],label = 'DFSM-DIY')
        
        ax.set_title(control,fontsize = fontsize_axlabel)
        ax.set_xlim(tspan)
        ax.tick_params(labelsize=fontsize_tick)
        ax.legend(ncol = ncol,fontsize = fontsize_legend)
        ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
        
        if save_flag:
            if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ control +'_comp.pdf')

    #------------------------------------------------------
    # Plot States
    #------------------------------------------------------
    for ix,state in enumerate(reqd_states[:3]):
            
        fig,ax = plt.subplots(1)
        ax.plot(time_of,states_of[:,ix],label = 'OpenFAST')
        ax.plot(time_dfsm,states_dfsm[:,ix],label = 'DFSM')

        if diy_flag:
            ax.plot(time_dfsm2,states_dfsm2[:,ix],label = 'DFSM-DIY')
        
        ax.set_title(state,fontsize = fontsize_axlabel)
        ax.set_xlim(tspan)
        ax.tick_params(labelsize=fontsize_tick)
        ax.legend(ncol = ncol,fontsize = fontsize_legend)
        ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
        
        if save_flag:
            if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ state +'_comp.pdf')

    #------------------------------------------
    # Plot Outputs
    #------------------------------------------
    for iy,output in enumerate(reqd_outputs):

        fig,ax = plt.subplots(1)
        ax.plot(time_of,outputs_of[:,iy],label = 'OpenFAST')
        ax.plot(time_dfsm,outputs_dfsm[:,iy],label = 'DFSM')

        if diy_flag:
            ax.plot(time_dfsm2,outputs_dfsm2[:,iy],label = 'DFSM-DIY')
        
        
        ax.set_title(output,fontsize = fontsize_axlabel)
        ax.set_xlim(tspan)
        ax.tick_params(labelsize=fontsize_tick)
        ax.legend(ncol = ncol,fontsize = fontsize_legend)
        ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
        
        if save_flag:
            if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ output + '_comp.pdf')

 
    plt.show()
    
        
        
        
        
        


