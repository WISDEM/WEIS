import matplotlib.pyplot as plt 
import numpy as np
import os, platform
import sys

# ROSCO toolbox modules 
from rosco.toolbox import control_interface as ROSCO_ci
from rosco.toolbox.inputs.validation import load_rosco_yaml
from scipy.interpolate import CubicSpline,interp1d


import time as timer

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension
from weis.dfsm.construct_dfsm import DFSM


from weis.dfsm.ode_algorithms import RK4


    

if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # parameter file
    parameter_filename = os.path.join(this_dir,'IEA_15_MW','IEA15MW-tuning.yaml')
    
    # tune directory
    
    # load
    inps = load_rosco_yaml(parameter_filename)
    path_params         = inps['path_params']
    turbine_params      = inps['turbine_params']
    controller_params   = inps['controller_params']
    
    # path to DISCON library
    lib_name = os.path.join('/home/athulsun/anaconda3/envs/weis-env/lib','libdiscon.so')

    # Write parameter input file
    param_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_DISCON.IN')
    
    # datapath
    region = 'LPV'
    datapath =  '/home/athulsun/DFSM/data' + os.sep + 'FOWT_1p6' 
    testpath = '/home/athulsun/DFSM/data' + os.sep + '1p6_test' 

    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)

    outfiles_test = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)
    
    # required states, controls and outputs
    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr'] 
    
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,1,100]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1])
                  }
    
    # filter parameters
    filter_args = {'state_filter_flag': [True,True,True],
                   'state_filter_type': [['filtfilt'],['filtfilt'],['filtfilt']],
                   'state_filter_tf': [[0.1],[0.1],[0.1]]
                   }
    
    # name of mat file that has the linearized models
    mat_file_name = this_dir + os.sep + 'FOWT_1p6_LPV.mat'
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    sim_detail_test = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    save_path = 'plots_ROSCO'
    # load and process data
    sim_detail.load_openfast_sim()
    sim_detail_test.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail_test.FAST_sim
    
    # split of training-testing data
    n_samples = 1
    test_inds = [16,18]
    
    # construct surrogate model
    dfsm = DFSM(sim_detail,n_samples = n_samples,L_type = 'LPV',N_type = None, train_split = 0.5)
    dfsm.construct_surrogate()
    dfsm.simulation_time = []
        
    for idx,ind in enumerate(test_inds):
            
        test_data = FAST_sim[ind]

        # extract controls
        controls_of = test_data['controls']
        states_of = test_data['states']

        if len(reqd_outputs)>0:
            outputs_of = test_data['outputs']
        
        # extract wind speed
        wind_speed = controls_of[:,dfsm.wind_speed_ind] #
        nt = len(wind_speed)
        t0 = 0;tf = 600
        time_of = np.linspace(t0,tf,nt)
        
        # create interpolating function for 
        w_pp = CubicSpline(time_of, wind_speed)
        wind_fun = lambda t: w_pp(t)
        w0 = wind_fun(0)
        bp0 = 16
        gt0 = 19000 
        
        if reqd_controls[-1] == 'Wave1Elev':
            wave_elev = controls_of[:,dfsm.wave_elev_ind]
            wave_pp = CubicSpline(time_of,wave_elev)
            wave_fun = lambda t:wave_pp(t)

        else:
            wave_fun = None
        
        # time span
        tspan = [t0,tf]
        
        # initial states
        x0 = test_data['states'][0,:]
        
        # parameters for ROSCO
        dt = 0.01
        num_blade = int(3)
        args = {'DT': dt, 'num_blade': int(3)}

        # Load controller library
        controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)
        
        if True: 
            # hardcoded for now
            param = {'VS_GenEff':95.89835000000,
                     'WE_GearboxRatio':1.0,
                     'VS_RtPwr':15000000.00000,
                     'time':[t0],
                     'dt':[dt],
                     'blade_pitch':[bp0],
                     'gen_torque':[gt0],
                     't0':t1,
                     'tf':tf,
                     'w_fun':wind_fun,
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
            
            # kill controller
            param['controller_interface'].kill_discon()
            

            #----------------------------------------------------------------------
            # Plot results
            #----------------------------------------------------------------------
            save_flag = False;plot_path = 'plots_DFSM_1p6'

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
                
                ax.set_title(control,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ control +str(idx) +'_comp.pdf')

            #------------------------------------------------------
            # Plot States
            #------------------------------------------------------
            
            for ix,state in enumerate(reqd_states):
                 
                fig,ax = plt.subplots(1)
                ax.plot(time_of,states_of[:,iu],label = 'OpenFAST')
                ax.plot(time_dfsm,states_dfsm[:,iu],label = 'DFSM')
                
                ax.set_title(state,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ state +str(idx) +'_comp.pdf')

            
                
            #------------------------------------------
            # Plot Outputs
            #------------------------------------------
            
            if dfsm.n_outputs > 0:

                for iy in range(reqd_outputs):

                    fig,ax = plt.subplots(1)
                    ax.plot(time_of,outputs_of[:,iy],label = 'OpenFAST')
                    ax.plot(time_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
                    
                    
                    ax.set_title(reqd_outputs[iy],fontsize = fontsize_axlabel)
                    ax.set_xlim(tspan)
                    ax.tick_params(labelsize=fontsize_tick)
                    ax.legend(ncol = 2,fontsize = fontsize_legend)
                    ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                    
                    if save_flag:
                        if not os.path.exists(plot_path):
                                os.makedirs(plot_path)
                            
                        fig.savefig(plot_path +os.sep+ reqd_outputs[iy] +str(idx) + '_comp.pdf')


    plt.show()
    
        
        
        
        
        


