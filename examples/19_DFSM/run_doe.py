import os 
import sys 
import numpy as np
import pickle
from scipy.interpolate import CubicSpline
import shutil
import matplotlib.pyplot as plt
import fatpack

# ROSCO toolbox modules 
from rosco.toolbox import control_interface as ROSCO_ci
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox.utilities import write_DISCON
from rosco import discon_lib_path
from rosco.toolbox.inputs.validation import load_rosco_yaml


# DFSM modules
from weis.dfsm.ode_algorithms import RK4,ABM4


# WEIS specific modules
import weis.inputs as sch
import time as timer


this_dir = os.path.dirname(os.path.realpath(__file__))
weis_dir = os.path.dirname(os.path.dirname(this_dir))

def extract_result(kp_dir,iter,state_names,control_names,output_names):

    pkl_file = kp_dir + str(iter)+ os.sep +'timeseries'+os.sep +'weis_job_0.p'

    with open(pkl_file,'rb') as handle:
        openfast_outputs = pickle.load(handle)

    test = {}

    test['Time'] = np.array(openfast_outputs['Time'])
    
    for state in state_names:
        if not(state[0] == 'd'):
            test[state] = np.array(openfast_outputs[state])

    for control in control_names:
        test[control] = np.array(openfast_outputs[control])

    for output in output_names:
        test[output] = np.array(openfast_outputs[output])
              

    return test


if __name__ == '__main__':


    # read WEIS options:
    mydir                       = this_dir  # get path to this file

    kp_dir = this_dir + os.sep +'outputs'+os.sep+ 'doe_kp' +os.sep + 'openfast_runs' + os.sep + 'iteration_'

    t_transient = 200

    plot_path = 'plot_doe';save_flag = True
    discon_folder = 'KP'

    #-----------------------------------------------------------------------------------------

    # pickle with the saved DFSM model
    pkl_name = this_dir + os.sep +'dfsm_1p6_python.pkl'

    # load dfsm model
    with open(pkl_name,'rb') as handle:
        dfsm = pickle.load(handle)

    interp_type = 'nearest'
    # dfsm.setup_LPV(interp_type)
    dfsm.simulation_time = []

    n_test = 5

    state_names = ['PtfmPitch','TTDspFA','GenSpeed','dPtfmPitch','dTTDspFA','dGenSpeed'];ns = len(state_names)
    control_names = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    output_names = ['TwrBsFxt', 'TwrBsMyt', 'YawBrTAxp', 'NcIMURAys', 'GenPwr','RtFldCp','RtFldCt']

    #------------------------------------------------------------------------------------------

    kp_list = np.linspace(0,40,n_test)

    # Write parameter input file
    cp_filename = os.path.join(this_dir,discon_folder,'weis_job_0_Cp_Ct_Cq.txt')
    
    tspan = [0,800]


    genspeed_max = np.zeros((n_test,2))
    ptfmpitch_max = np.zeros((n_test,2))
    DEL = np.zeros((n_test,2))
    omega_all = np.zeros((n_test,))

    # parameters
    bins = 10
    load2stress = 1
    slope = 4
    ult_stress = 4
    s_intercept = 1
    elapsed = 800 - t_transient


    for i in range(n_test):

        test = extract_result(kp_dir,i,state_names,control_names,output_names)
        discon_file = 'weis_job_'+str(i)+'_DISCON.IN'
        param_filename = os.path.join(this_dir,discon_folder,discon_file)

        # path to DISCON library
        lib_name = '/home/athulsun/anaconda3/envs/weis-dev/lib/libdiscon.so'
        

        tt = test['Time']-00
        t0 = tt[0]
        tf = tt[-1]
        wind_speed = test['RtVAvgxh']
        wave_elev = test['Wave1Elev']

        wind_fun = CubicSpline(tt,wind_speed)
        wave_fun = CubicSpline(tt,wave_elev)

        bp0 = test['BldPitch1'][0]
        gt0 = test['GenTq'][0]

        x0 = np.zeros((ns,))

        # hardcoded for now
        dt = 0.01
        num_blade = int(2)

        args = {'DT': dt, 'num_blade': num_blade,'pitch':bp0}

        # hardcoded for now
        param = {'VS_GenEff':95.75622000000,
                    'WE_GearboxRatio':1.0,
                    'VS_RtPwr':95.75622000000,
                    'time':[t0],
                    'dt':[dt],
                    'blade_pitch':[bp0],
                    'gen_torque':[gt0],
                    't0':t0,
                    'tf':tf,
                    'w_fun':wind_fun,
                    'gen_speed_scaling':1,
                    'wave_fun':wave_fun,
                    'ny': len(output_names),
                    'args':args,
                    'param_filename':param_filename,
                    'lib_name':lib_name,
                    'controller_interface' : ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test',**args)
                    }
        
        # start timer and solve for the states and controls
        t1 = timer.time()
        time_dfsm,states_dfsm,controls_dfsm,outputs_dfsm = RK4(x0, dt, tspan, dfsm, param)
        t2 = timer.time()
        dfsm.simulation_time.append(t2-t1)

        dt_of = tt[1] - tt[0]

        if not(dt == dt_of):
            states_dfsm = CubicSpline(time_dfsm,states_dfsm)(tt)
            controls_dfsm = CubicSpline(time_dfsm,controls_dfsm)(tt)
            outputs_dfsm = CubicSpline(time_dfsm,outputs_dfsm)(tt)
            time_dfsm = tt

        t_ind = time_dfsm >= t_transient 
        
        time_dfsm = time_dfsm[t_ind]
        time_dfsm = time_dfsm - t_transient
        states_dfsm = states_dfsm[t_ind,:]
        controls_dfsm = controls_dfsm[t_ind,:]
        outputs_dfsm = outputs_dfsm[t_ind,:]
        
        # shutdown controller
        param['controller_interface'].kill_discon()
        tspan_ = [0,600]
        
        #----------------------------------------------------------------------
        # Plot results
        #----------------------------------------------------------------------

        # plot properties
        markersize = 10
        linewidth = 1.5
        fontsize_legend = 16
        fontsize_axlabel = 12
        fontsize_tick = 12
        #tspan_ = [time_dfsm[0],time_dfsm[-1]]


        #------------------------------------------------
        # Plot controls
        #-----------------------------------------------
        
        for iu,control in enumerate(control_names):
                
            fig,ax = plt.subplots(1)

            # if control == 'RtVAvgxh':
            #     control = 'Wind' 

            # if control == 'Wave1Elev':
            #     control = 'Wave'

            ax.plot(time_dfsm,test[control][t_ind],label = 'OpenFAST')
            ax.plot(time_dfsm,controls_dfsm[:,iu],label = 'DFSM')

            
            ax.set_title(control,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan_)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ control+'_' +str(i)+'_comp.pdf')

            plt.close(fig)

        #------------------------------------------------------
        # Plot States
        #------------------------------------------------------
        for ix,state in enumerate(state_names[:3]):
                
            fig,ax = plt.subplots(1)

            if state == 'GenSpeed':
                fac = 1

                genspeed_max[i,0]= np.max(np.array(test['GenSpeed']))
        # Load controller library
            if state == 'GenSpeed':
                genspeed_max[i,1] = np.max(states_dfsm[:,ix]*fac)
            else:
                fac = 1

            if state == 'PtfmPitch':
                 ptfmpitch_max[i,0] = np.max(np.array(test['PtfmPitch']))
                 ptfmpitch_max[i,1] = np.max(states_dfsm[:,ix])
            
            ax.plot(time_dfsm,test[state][t_ind],label = 'OpenFAST')
            ax.plot(time_dfsm,states_dfsm[:,ix]*fac,label = 'DFSM')
            
            ax.set_title(state,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan_)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ state +'_' +str(i)+'_comp.pdf')

            plt.close(fig)

        #------------------------------------------
        # Plot Outputs
        #------------------------------------------
        for iy,output in enumerate(output_names):

            if output == 'TwrBsMyt':
                TwrBsMyt_of = np.array(test['TwrBsMyt'][t_ind])
                TwrBsMyt_dfsm = outputs_dfsm[:,iy]

                F_of, Fmean_of = fatpack.find_rainflow_ranges(TwrBsMyt_of, return_means=True)
                Nrf_of, Frf_of = fatpack.find_range_count(F_of, bins)
                DELs_ = Frf_of ** slope * Nrf_of / elapsed
                DEL[i,0] = DELs_.sum() ** (1.0 / slope)

                F_dfsm, Fmean_dfsm = fatpack.find_rainflow_ranges(TwrBsMyt_dfsm, return_means=True)
                Nrf_dfsm, Frf_dfsm = fatpack.find_range_count(F_dfsm, bins)
                DELs_ = Frf_dfsm ** slope * Nrf_dfsm / elapsed
                DEL[i,1] = DELs_.sum() ** (1.0 / slope)

            fig,ax = plt.subplots(1)
            
            ax.plot(time_dfsm,test[output][t_ind],label = 'OpenFAST')
            ax.plot(time_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
            
            
            ax.set_title(output,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan_)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ output +'_' +str(i)+ '_comp.pdf')

            plt.close(fig)

    arr_shape = [1,n_test]

    omega_all = np.reshape(omega_all,arr_shape)

    genspeed_max_of = genspeed_max[:,0]
    genspeed_max_dfsm = genspeed_max[:,1]
    genspeed_max_of = np.reshape(genspeed_max_of,arr_shape)
    genspeed_max_dfsm = np.reshape(genspeed_max_dfsm,arr_shape)
    genspeed_max_of = np.mean(genspeed_max_of,axis = 0)
    genspeed_max_dfsm = np.mean(genspeed_max_dfsm,axis = 0)
    
    ptfmpitch_max_of = ptfmpitch_max[:,0]
    ptfmpitch_max_dfsm = ptfmpitch_max[:,1]
    ptfmpitch_max_of = np.reshape(ptfmpitch_max_of,arr_shape)
    ptfmpitch_max_dfsm = np.reshape(ptfmpitch_max_dfsm,arr_shape)
    ptfmpitch_max_of = np.mean(ptfmpitch_max_of,axis = 0)
    ptfmpitch_max_dfsm = np.mean(ptfmpitch_max_dfsm,axis = 0)

    DEL_of = DEL[:,0]
    DEL_dfsm = DEL[:,1]
    DEL_of = np.reshape(DEL_of,arr_shape)
    DEL_dfsm = np.reshape(DEL_dfsm,arr_shape)
    DEL_of = np.mean(DEL_of,axis = 0)
    DEL_dfsm = np.mean(DEL_dfsm,axis = 0)


    

    print(np.mean(np.array(dfsm.simulation_time)))
    #breakpoint()

    fig,ax = plt.subplots(1)
    ax.plot(kp_list,genspeed_max_of,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(kp_list,genspeed_max_dfsm,'.-',label = 'DFSM',markersize = markersize)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel+3)
    ax.set_ylabel('Max. GenSpeed [rpm]',fontsize = fontsize_axlabel+3)
    ax.legend(ncol = 2,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'genspeed_max.pdf')

    plt.close(fig)


    fig,ax = plt.subplots(1)
    ax.plot(kp_list,ptfmpitch_max_of,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(kp_list,ptfmpitch_max_dfsm,'.-',label = 'DFSM',markersize = markersize)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel+3)
    ax.set_ylabel('Max. PtfmPitch [deg]',fontsize = fontsize_axlabel+3)
    ax.legend(ncol = 2,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'ptfmpitch_max.pdf')
    plt.close(fig)


    fig,ax = plt.subplots(1)
    ax.plot(kp_list,DEL_of/1e4,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(kp_list,DEL_dfsm/1e4,'.-',label = 'DFSM',markersize = markersize)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel+3)
    ax.set_ylabel('DEL_twrbsmyt [kNm]',fontsize = fontsize_axlabel+3)
    ax.legend(ncol = 2,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'DEL.pdf')
    plt.close(fig)
    
    

    results_dict = {'omega_pc_list':kp_list,
                    'genspeed_max_of':genspeed_max_of,
                    'genspeed_max_dfsm':genspeed_max_dfsm,
                    'ptfmpitch_max_of':ptfmpitch_max_of,
                    'ptfmpitch_max_dfsm':ptfmpitch_max_dfsm,
                    'DEL_of':DEL_of/1e4,
                    'DEL_dfsm':DEL_dfsm/1e4}
    
    results_file = 'doe_comparison_results.pkl'

    # with open(results_file,'wb') as handle:
    #     pickle.dump(results_dict,handle)

    # plt.show()
    #breakpoint()


    










