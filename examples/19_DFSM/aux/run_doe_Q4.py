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
from weis.dfsm.ode_algorithms import RK4


# WEIS specific modules
import weis.inputs as sch
from weis.aeroelasticse.turbsim_util    import generate_wind_files
from weis.aeroelasticse.turbsim_file    import TurbSimFile
from weis.dlc_driver.dlc_generator      import DLCGenerator
from weis.dfsm.generate_wave_elev import generate_wave_elev
import time as timer


this_dir = os.path.dirname(os.path.realpath(__file__))
weis_dir = os.path.dirname(os.path.dirname(this_dir))


def generate_test_data(this_dir,modeling_options,wt_init):

    # Extract user defined list of cases
    DLCs = modeling_options['DLC_driver']['DLCs']

    # Initialize the generator
    metocean = modeling_options['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(
        metocean = metocean
    )

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)


    # generate wind files
    FAST_namingOut = 'oloc'
    wind_directory = os.path.join(this_dir,'oloc/wind')
    if not os.path.exists(wind_directory):
        os.makedirs(wind_directory)
    rotorD = wt_init['assembly']['rotor_diameter']
    hub_height = wt_init['assembly']['hub_height']

    # from various parts of openmdao_openfast:
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [''] * dlc_generator.n_cases

    test_dataset = []
    turbsim_exe = shutil.which('turbsim')

    #fig,ax = plt.subplots(2)
    
    rng = np.random.default_rng(12345)
    for i_case in range(dlc_generator.n_cases):
        dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time

        if dlc_generator.cases[i_case].IECturbc > 0:    # use custom TI for DLC case
                    dlc_generator.cases[i_case].IECturbc = str(dlc_generator.cases[i_case].IECturbc)
                    dlc_generator.cases[i_case].IEC_WindType = 'NTM'
        else:
            dlc_generator.cases[i_case].IECturbc = wt_init['assembly']['turbulence_class']

        WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
            dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height,turbsim_exe,i_case)

        # Compute rotor average wind speed as level2_disturbances
        ts_file     = TurbSimFile(WindFile_name[i_case])
        ts_file.compute_rot_avg(rotorD/2)
        u_h         = ts_file['rot_avg'][0,:]

        off = max(u_h) - 25
        ind = u_h > 25
        
        # remove any windspeeds > 25 m/s
        u_h[ind] = 25
        
        tt          = ts_file['t']

        
        eta = generate_wave_elev(tt,dlc_generator.cases[i_case].wave_height,dlc_generator.cases[i_case].wave_period,rng)

        test_dataset.append({'Time':tt, 'Wind': u_h, 'Wave': eta})


    return test_dataset,dlc_generator.cases[i_case].transient_time




if __name__ == '__main__':


    # read WEIS options:
    mydir                       = this_dir  # get path to this file

    # modeling options
    fname_modeling_options      = mydir + os.sep + "modeling_options_dfsm_fowt.yaml"
    modeling_options            = sch.load_modeling_yaml(fname_modeling_options)

    # geometry yaml
    fname_wt_input              = os.path.join(this_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
    wt_init                     = sch.load_geometry_yaml(fname_wt_input)

    # analysis options
    fname_analysis_options      = mydir + os.sep + "analysis_options_dfsm_fowt.yaml"
    analysis_options            = sch.load_analysis_yaml(fname_analysis_options)


    # generate test data set
    #test_dataset,t_transient = generate_test_data(this_dir,modeling_options,wt_init)
    pkl_name = 'fowt_omega_19.pkl'
    with open(pkl_name,'rb') as handle:
        test_dataset = pickle.load(handle)

    t_transient = 0

    plot_path = 'plot_doe_ssid';save_flag = True

    #-----------------------------------------------------------------------------------------

    # pickle with the saved DFSM model
    pkl_name = this_dir + os.sep +'dfsm_1p6.pkl'

    # load dfsm model
    with open(pkl_name,'rb') as handle:
        dfsm = pickle.load(handle)


    n_test = len(test_dataset)

    state_names = ['PtfmPitch','TTDspFA','GenSpeed','dPtfmPitch','dTTDspFA','dGenSpeed'];ns = len(state_names)
    control_names = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    output_names = ['TwrBsFxt', 'TwrBsMyt', 'YawBrTAxp', 'NcIMURAys', 'GenPwr']

    #------------------------------------------------------------------------------------------

    omega_pc_list = np.linspace(0.1,0.3,12)

    # Write parameter input file
    cp_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_Cp_Ct_Cq.txt')
    param_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_DISCON.IN')

    rosco_options = modeling_options['ROSCO']

    tuning_yaml = os.path.join(this_dir,'IEA_15_MW','IEA15MW-tuning.yaml')

    tt = np.array(test_dataset[0]['Time'])
    
    t0 = tt[0];tf = tt[-1]

    
    tspan = [t0,tf]
    dt = 0.01

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
    elapsed = tf

    


    for i in range(n_test):

        test = test_dataset[i]

        inputs = load_rosco_yaml(tuning_yaml)

        path_params         = inputs['path_params']
        turbine_params      = inputs['turbine_params']
        controller_params   = inputs['controller_params']

        controller_params['omega_pc'] = test['omega']
        omega_all[i] = test['omega']

        turbine = ROSCO_turbine.Turbine(turbine_params)
        turbine.v_rated = 10.555555555555555

        turbine.load_from_fast(
            path_params['FAST_InputFile'],
            os.path.join(this_dir,path_params['FAST_directory']),
            rot_source='txt',txt_filename=cp_filename
            )
        
        
        # Tune controller 
        controller      = ROSCO_controller.Controller(controller_params)
        controller.tune_controller(turbine)

        # Write parameter input file
        param_filename = os.path.join(this_dir,'DISCON_test.IN')
        write_DISCON(
        turbine,controller,
        param_file=param_filename, 
        txt_filename=cp_filename
        )


        # path to DISCON library
        lib_name = discon_lib_path
        

        tt = test['Time']
        wind_speed = test['Wind']
        wave_elev = test['Wave']

        wind_fun = CubicSpline(tt,wind_speed)
        wave_fun = CubicSpline(tt,wave_elev)

        bp0 = 5.82
        gt0 = 19941

        x0 = np.zeros((ns,))
        x0[2] = 7.490320205688477/100
        args = {'DT': dt, 'num_blade': int(3),'pitch':bp0}

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

        t_ind = time_dfsm >= t_transient 

        time_dfsm = time_dfsm[t_ind]
        time_dfsm = time_dfsm - t_transient
        states_dfsm = states_dfsm[t_ind,:]
        controls_dfsm = controls_dfsm[t_ind,:]
        outputs_dfsm = outputs_dfsm[t_ind,:]
        
        # shutdown controller
        param['controller_interface'].kill_discon()
        
        #----------------------------------------------------------------------
        # Plot results
        #----------------------------------------------------------------------

        # plot properties
        markersize = 10
        linewidth = 1.5
        fontsize_legend = 16
        fontsize_axlabel = 12
        fontsize_tick = 12
        tspan_ = [time_dfsm[0],time_dfsm[-1]]

        # fig,ax = plt.subplots(3)

        # ax[0].plot(time_dfsm,controls_dfsm[:,0])
        # ax[0].set_ylabel('RtVAvgxh',fontsize = fontsize_axlabel)
        # ax[0].set_xlim(tspan_)
        # ax[0].tick_params(labelsize=fontsize_tick)
        # ax[0].set_xlabel('Time [s]',fontsize = fontsize_axlabel)

        # ax[1].plot(time_dfsm,controls_dfsm[:,1]/1000)
        # ax[1].set_ylabel('GenTq',fontsize = fontsize_axlabel)
        # ax[1].set_xlim(tspan_)
        # ax[1].tick_params(labelsize=fontsize_tick)
        # ax[1].set_xlabel('Time [s]',fontsize = fontsize_axlabel)

        # ax[2].plot(time_dfsm,controls_dfsm[:,2])
        # ax[2].set_ylabel('BldPitch1',fontsize = fontsize_axlabel)
        # ax[2].set_xlim(tspan_)
        # ax[2].tick_params(labelsize=fontsize_tick)
        # ax[2].set_xlabel('Time [s]',fontsize = fontsize_axlabel)

        # w = int(np.ceil(np.mean(controls_dfsm[:,0])))

        # fig.subplots_adjust(hspace = 0.65)

        # if save_flag:
        #     if not os.path.exists(plot_path):
        #             os.makedirs(plot_path)
                
        #     fig.savefig(plot_path +os.sep+'controls_vplot_'+str(w)+'_' +str(i) + '.pdf')

        #------------------------------------------------
        # Plot controls
        #-----------------------------------------------
        
        for iu,control in enumerate(control_names):
                
            fig,ax = plt.subplots(1)

            if control == 'RtVAvgxh':
                control = 'Wind' 

            if control == 'Wave1Elev':
                control = 'Wave'

            ax.plot(test['Time'],test[control],label = 'OpenFAST')
            ax.plot(time_dfsm,controls_dfsm[:,iu],label = 'DFSM')

            
            ax.set_title(control,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ control+'_' +str(i)+'_comp.pdf')

            plt.close()

        #------------------------------------------------------
        # Plot States
        #------------------------------------------------------
        for ix,state in enumerate(state_names[:3]):
                
            fig,ax = plt.subplots(1)

            if state == 'GenSpeed':
                fac = 100

                genspeed_max[i,0] = np.max(np.array(test['GenSpeed']))
                genspeed_max[i,1] = np.max(states_dfsm[:,ix]*fac)
            else:
                fac = 1

            if state == 'PtfmPitch':
                 ptfmpitch_max[i,0] = np.max(np.array(test['PtfmPitch']))
                 ptfmpitch_max[i,1] = np.max(states_dfsm[:,ix])
            
            ax.plot(test['Time'],test[state],label = 'OpenFAST')
            ax.plot(time_dfsm,states_dfsm[:,ix]*fac,label = 'DFSM')
            
            ax.set_title(state,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ state +'_' +str(i)+'_comp.pdf')

            plt.close()

        #------------------------------------------
        # Plot Outputs
        #------------------------------------------
        for iy,output in enumerate(output_names):

            if output == 'TwrBsMyt':
                TwrBsMyt_of = np.array(test['TwrBsMyt'])
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
            
            ax.plot(test['Time'],test[output],label = 'OpenFAST')
            ax.plot(time_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
            
            
            ax.set_title(output,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ output +'_' +str(i)+ '_comp.pdf')

            plt.close()

    

    omega_all = np.reshape(omega_all,[10,12])

    genspeed_max_of = genspeed_max[:,0]
    genspeed_max_dfsm = genspeed_max[:,1]
    genspeed_max_of = np.reshape(genspeed_max_of,[10,12])
    genspeed_max_dfsm = np.reshape(genspeed_max_dfsm,[10,12])
    genspeed_max_of = np.mean(genspeed_max_of,axis = 0)
    genspeed_max_dfsm = np.mean(genspeed_max_dfsm,axis = 0)
    
    ptfmpitch_max_of = ptfmpitch_max[:,0]
    ptfmpitch_max_dfsm = ptfmpitch_max[:,1]
    ptfmpitch_max_of = np.reshape(ptfmpitch_max_of,[10,12])
    ptfmpitch_max_dfsm = np.reshape(ptfmpitch_max_dfsm,[10,12])
    ptfmpitch_max_of = np.mean(ptfmpitch_max_of,axis = 0)
    ptfmpitch_max_dfsm = np.mean(ptfmpitch_max_dfsm,axis = 0)

    DEL_of = DEL[:,0]
    DEL_dfsm = DEL[:,1]
    DEL_of = np.reshape(DEL_of,[10,12])
    DEL_dfsm = np.reshape(DEL_dfsm,[10,12])
    DEL_of = np.mean(DEL_of,axis = 0)
    DEL_dfsm = np.mean(DEL_dfsm,axis = 0)


    

    print(np.mean(np.array(dfsm.simulation_time)))
    #breakpoint()

    fig,ax = plt.subplots(1)
    ax.plot(omega_pc_list,genspeed_max_of,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(omega_pc_list,genspeed_max_dfsm,'.-',label = 'DFSM',markersize = markersize)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel+3)
    ax.set_ylabel('Max. GenSpeed [rpm]',fontsize = fontsize_axlabel+3)
    ax.legend(ncol = 2,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'genspeed_max.pdf')


    fig,ax = plt.subplots(1)
    ax.plot(omega_pc_list,ptfmpitch_max_of,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(omega_pc_list,ptfmpitch_max_dfsm,'.-',label = 'DFSM',markersize = markersize)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel+3)
    ax.set_ylabel('Max. PtfmPitch [deg]',fontsize = fontsize_axlabel+3)
    ax.legend(ncol = 2,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'ptfmpitch_max.pdf')


    fig,ax = plt.subplots(1)
    ax.plot(omega_pc_list,DEL_of/1e4,'.-',label = 'OpenFAST',markersize = markersize)
    ax.plot(omega_pc_list,DEL_dfsm/1e4,'.-',label = 'DFSM',markersize = markersize)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_xlabel('Omega_PC [rad/s]',fontsize = fontsize_axlabel+3)
    ax.set_ylabel('DEL_twrbsmyt [kNm]',fontsize = fontsize_axlabel+3)
    ax.legend(ncol = 2,fontsize = fontsize_legend)

    if save_flag:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            
        fig.savefig(plot_path +os.sep+ 'DEL.pdf')
    
    

    results_dict = {'omega_pc_list':omega_pc_list,
                    'genspeed_max_of':genspeed_max_of,
                    'genspeed_max_dfsm':genspeed_max_dfsm,
                    'ptfmpitch_max_of':ptfmpitch_max_of,
                    'ptfmpitch_max_dfsm':ptfmpitch_max_dfsm,
                    'DEL_of':DEL_of/1e4,
                    'DEL_dfsm':DEL_dfsm/1e4}
    
    results_file = 'doe_comparison_results.pkl'

    with open(results_file,'wb') as handle:
        pickle.dump(results_dict,handle)

    plt.show()
    #breakpoint()


    










