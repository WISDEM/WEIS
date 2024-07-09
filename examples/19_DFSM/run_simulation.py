'''
Example to run a close-loop simulation with the DFSM model used as 'plant' for the ROSCO controller.
'''

import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os
import time as timer
from scipy.interpolate import CubicSpline

# ROSCO toolbox modules 
from rosco.toolbox import control_interface as ROSCO_ci
from rosco import discon_lib_path

# DFSM modules
from weis.dfsm.ode_algorithms import RK4

if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # path to DISCON library
    lib_name = discon_lib_path

    # Write parameter input file
    param_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_DISCON.IN')
    
    plot_path = 'plot_dfsm_result';save_flag = False

    # pickle with the saved DFSM model
    pkl_name = this_dir + os.sep +'dfsm_1p6.pkl'

    # load dfsm model
    with open(pkl_name,'rb') as handle:
        dfsm = pickle.load(handle)

    # pickle file with the test data
    test_pkl_name = this_dir + os.sep +'test_data_1p6.pkl'

    # load test data
    with open(test_pkl_name,'rb') as handle:
        FAST_sim = pickle.load(handle)
    
    test_data = FAST_sim[0]


    # extract controls
    controls_of = test_data['controls']
    states_of = test_data['states']
    outputs_of = test_data['outputs']

    # get control, state, output names
    state_names = test_data['state_names'];ns = len(state_names)
    control_names = test_data['control_names']
    output_names = test_data['output_names']
    
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
    
    if control_names[-1] == 'Wave1Elev':
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
    
    # hardcoded for now
    param = {'VS_GenEff':95.89835000000,
                'WE_GearboxRatio':1.0,
                'VS_RtPwr':15000000.00000,
                'time':[t0],
                'dt':[dt],
                'blade_pitch':[bp0],
                'gen_torque':[gt0],
                't0':t0,
                'tf':tf,
                'w_fun':wind_fun,
                'gen_speed_scaling':100,
                'controller_interface':controller_interface,
                'wave_fun':wave_fun,
                'ny': len(output_names)
                }
        
    # start timer and solve for the states and controls
    t1 = timer.time()
    time_dfsm, states_dfsm, controls_dfsm,outputs_dfsm,T_extrap, U_extrap = RK4(x0, dt, tspan, dfsm, param)
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
    for iu,control in enumerate(control_names):
            
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
                
            fig.savefig(plot_path +os.sep+ control +'_comp.pdf')

    #------------------------------------------------------
    # Plot States
    #------------------------------------------------------
    for ix,state in enumerate(state_names[:3]):
            
        fig,ax = plt.subplots(1)
        ax.plot(time_of,states_of[:,ix],label = 'OpenFAST')
        ax.plot(time_dfsm,states_dfsm[:,ix],label = 'DFSM')
        
        ax.set_title(state,fontsize = fontsize_axlabel)
        ax.set_xlim(tspan)
        ax.tick_params(labelsize=fontsize_tick)
        ax.legend(ncol = 2,fontsize = fontsize_legend)
        ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
        
        if save_flag:
            if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ state +'_comp.pdf')

    #------------------------------------------
    # Plot Outputs
    #------------------------------------------
    for iy,output in enumerate(output_names):

        fig,ax = plt.subplots(1)
        ax.plot(time_of,outputs_of[:,iy],label = 'OpenFAST')
        ax.plot(time_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
        
        
        ax.set_title(output,fontsize = fontsize_axlabel)
        ax.set_xlim(tspan)
        ax.tick_params(labelsize=fontsize_tick)
        ax.legend(ncol = 2,fontsize = fontsize_legend)
        ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
        
        if save_flag:
            if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ output + '_comp.pdf')


    plt.show()
    
        
        
        
        
        


