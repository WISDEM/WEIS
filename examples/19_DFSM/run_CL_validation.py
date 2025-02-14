import matplotlib.pyplot as plt 
import numpy as np
import os, platform
import sys
import pickle
import fatpack

# ROSCO toolbox modules 
from rosco.toolbox import control_interface as ROSCO_ci
from rosco import discon_lib_path
from scipy.interpolate import CubicSpline
import time as timer
from rosco.toolbox.ofTools.util import spectral
from weis.aeroelasticse.CaseGen_General import case_naming

# DFSM modules
from weis.dfsm.ode_algorithms import RK4,ABM4
from scipy.interpolate import interp1d
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension,calculate_MSE,compile_dfsm_results

from weis.glue_code.mpi_tools import MPI

from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
from pCrunch.io import OpenFASTOutput

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 12

# parameters
bins = 10
slope = 4

magnitude_channels = {'TwrBsM': [ 'TwrBsMyt']}
fatigue_channels = {'TwrBsM': FatigueParams(slope=4)}


def evaluate_multi(case_data):

    # initialize controller interface
    case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])

    # run simulation
    t1 = timer.time()

    # run simulation
    if case_data['ode_algorithm'] == 'RK4':
        output = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
    elif case_data['ode_algorithm'] == 'ABM4':
        output = ABM4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

    t2 = timer.time()
    print(t2-t1)
    
    # shut down controller
    case_data['param']['controller_interface'].kill_discon()


    return output


def run_serial(case_data_all):

    # initialize storage array for results
    outputs = []

    # loop through and evaluate simulations    
    for case_data in case_data_all:
        
        # initialize controller interface
        case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])
        
        # run simulation
        if case_data['ode_algorithm'] == 'RK4':
            t,x,u,y = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
        elif case_data['ode_algorithm'] == 'ABM4':
            t,x,u,y = ABM4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
        
        # shut down controller
        case_data['param']['controller_interface'].kill_discon()

        # initialize
        sim_results = {}
        sim_results['case_no'] = case_data['case']
        sim_results['T_dfsm'] = t 
        sim_results['states_dfsm'] = x 
        sim_results['controls_dfsm'] = u 
        sim_results['outputs_dfsm'] = y

        outputs.append(sim_results)
    
    return outputs


def run_mpi(case_data_all,mpi_options):

    from mpi4py import MPI

    # mpi comm management
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sub_ranks = mpi_options['mpi_comm_map_down'][rank]

    size = len(sub_ranks)

    N_cases = len(case_data_all)
    N_loops = int(np.ceil(float(N_cases)/float(size)))

    sim_results = []
    for i in range(N_loops):
        idx_s    = i*size
        idx_e    = min((i+1)*size, N_cases)

        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            data   = [evaluate_multi, case_data]
            rank_j = sub_ranks[j]
            comm.send(data, dest=rank_j, tag=0)

        # for rank_j in sub_ranks:
        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            rank_j = sub_ranks[j]
            data_out = comm.recv(source=rank_j, tag=1)
            sim_results.append(data_out)

    # compile and store results
    outputs = []
    
    for icase,result_ in enumerate(sim_results):
        t = result_[0]
        x = result_[1]
        u = result_[2]
        y = result_[3]

        # initialize
        sim_results = {}

        # store
        sim_results['case_no'] = icase
        sim_results['T_dfsm'] = t 
        sim_results['states_dfsm'] = x 
        sim_results['controls_dfsm'] = u 
        sim_results['outputs_dfsm'] = y

        outputs.append(sim_results)



    return outputs




def run_closed_loop_simulation(dfsm,FAST_sim,dt,ode_algorithm,test_datapath,transition_time,save_flag,plot_path,PSD_path,format,test_inds,mpi_options):

    # path to DISCON library
    lib_name = discon_lib_path #'/home/athulsun/WEIS-AKS/local/lib/libdiscon.so'
     
    # Write parameter input file
    param_filename = test_datapath + os.sep +'RM1_00_DISCON.IN'

    case_names_dfsm = case_naming(len(FAST_sim),'DFSM')
    case_names_of = case_naming(len(FAST_sim),'OpenFAST')
   

    T_of = []
    X_OF_list = []
    U_OF_list = []
    Y_OF_list = []

    case_data_all = []

    for idx,ind in enumerate(test_inds):
            
        test_data = FAST_sim[ind]
        X_OF_list.append(test_data['states'])
        U_OF_list.append(test_data['controls'])
        Y_OF_list.append(test_data['outputs'])
        
        wind_speed = test_data['controls'][:,0] #
        nt = len(wind_speed)
        t0 = 0;tf = 800
        time_of = np.linspace(t0,tf,nt)

        T_of.append(time_of)
        
        w_fun = CubicSpline(time_of, wind_speed)
        gt0 = test_data['controls'][0,1]
        bp0 = test_data['controls'][0,2]
        
        if reqd_controls[-1] == 'Wave1Elev':
            wave_elev = test_data['controls'][:,3]
            wave_fun = CubicSpline(time_of,wave_elev)
        else:
            wave_fun = None
        
        # time span
        tspan = [t0,tf]
        
        x0 = test_data['states'][0,:]
        
        num_blade = int(2)

        args = {'DT': dt, 'num_blade': num_blade,'pitch':bp0}

        # hardcoded for now
        param = {'VS_GenEff':94.40000000000,
                    'WE_GearboxRatio':53.0,
                    'VS_RtPwr':500000.0000000,
                    'time':[t0],
                    'dt':[dt],
                    'blade_pitch':[bp0],
                    'gen_torque':[gt0],
                    't0':t0,
                    'tf':tf,
                    'w_fun':w_fun,
                    'gen_speed_scaling':100,
                    'wave_fun':wave_fun,
                    'ny': len(reqd_outputs),
                    'args':args,
                    'param_filename':param_filename,
                    'lib_name':lib_name
                    }
            
        case_data = {}
        case_data['case'] = idx
        case_data['param'] = param 
        case_data['dt'] = dt 
        case_data['x0'] = x0 
        case_data['tspan'] = tspan 
        case_data['dfsm'] = dfsm
        case_data['ode_algorithm'] = ode_algorithm

        case_data_all.append(case_data)

    if mpi_options['mpi_run']:

        # evaluate the closed loop simulations in parallel using MPI

        sim_results = run_mpi(case_data_all,mpi_options)

    else:

        # evaluate the closed loop simulations serially
        
        sim_results = run_serial(case_data_all)


    PSD_quantities = ['GenSpeed','TwrBsMyt','PtfmPitch','BldPitch1','GenTq','Wave1Elev']

    #-----------------------------------------------
    # Plot results
    #-----------------------------------------------

    
    elapsed = tf - transition_time
    tf = tf - transition_time
    tspan = [t0,tf]

    outlist_dfsm = []
    outlist_of = []

    for icase,sim_result in enumerate(sim_results):

        # extract results

        time_of = T_of[icase]
        time_dfsm = sim_result['T_dfsm']

        states_dfsm = sim_result['states_dfsm']
        states_of = X_OF_list[icase]

        controls_dfsm = sim_result['controls_dfsm']
        controls_of = U_OF_list[icase]

        outputs_dfsm = sim_result['outputs_dfsm']
        outputs_of = Y_OF_list[icase]

        dt_of = time_of[1] - time_of[0]

        if not(dt == dt_of):
            states_dfsm = CubicSpline(time_dfsm,states_dfsm)(time_of)
            controls_dfsm = CubicSpline(time_dfsm,controls_dfsm)(time_of)
            outputs_dfsm = CubicSpline(time_dfsm,outputs_dfsm)(time_of)
            time_dfsm = time_of
        

        t_ind = (time_dfsm >= transition_time)
        time_of = time_of[t_ind]; time_of = time_of - transition_time
        time_dfsm = time_dfsm[t_ind]; time_dfsm = time_dfsm - transition_time 

        states_dfsm = states_dfsm[t_ind,:];states_of = states_of[t_ind,:]
        controls_dfsm = controls_dfsm[t_ind,:];controls_of = controls_of[t_ind,:]
        outputs_dfsm = outputs_dfsm[t_ind,:];outputs_of = outputs_of[t_ind,:]

        #------------------------------------------------------
        # Plot Controls and Inputs
        #------------------------------------------------------

        for iu,control in enumerate(reqd_controls):

            if control in PSD_quantities:
                xf,FFT_of,_ = spectral.fft_wrap(time_of,controls_of[:,iu],averaging = 'Welch',averaging_window= 'hamming')
                xf,FFT_dfsm,_ = spectral.fft_wrap(time_dfsm,controls_dfsm[:,iu],averaging = 'Welch',averaging_window= 'hamming')

                fig1,ax1 = plt.subplots(1)
                ax1.loglog(xf,np.sqrt(FFT_of),label = 'OpenFAST')
                ax1.loglog(xf,np.sqrt(FFT_dfsm),label = 'DFSM')
                ax1.set_xlabel('Freq [Hz]',fontsize = fontsize_axlabel)
                ax1.set_title(control + '_PSD',fontsize = fontsize_axlabel)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                #ax1.set_xlim([np.min(xf),np.max(xf)])

                if save_flag:       
                    fig1.savefig(PSD_path +os.sep+ control + '_' + str(icase) +'_comp' + format)
                plt.close(fig1)

        
            fig,ax = plt.subplots(1)
            ax.plot(time_of,controls_of[:,iu],label = 'OpenFAST')
            ax.plot(time_dfsm,controls_dfsm[:,iu],label = 'DFSM')
            
            ax.set_title(control,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:       
                fig.savefig(plot_path +os.sep+ control + '_' + str(icase) +'_comp' + format)

            plt.close(fig)

            

        #------------------------------------------------------
        # Plot States
        #------------------------------------------------------
        for ix,state in enumerate(reqd_states):

            if state in PSD_quantities:
                xf,FFT_of,_ = spectral.fft_wrap(time_of,states_of[:,ix],averaging = 'Welch',averaging_window= 'hamming')
                xf,FFT_dfsm,_ = spectral.fft_wrap(time_dfsm,states_dfsm[:,ix],averaging = 'Welch',averaging_window= 'hamming')

                fig1,ax1 = plt.subplots(1)
                ax1.loglog(xf,np.sqrt(FFT_of),label = 'OpenFAST')
                ax1.loglog(xf,np.sqrt(FFT_dfsm),label = 'DFSM')
                ax1.set_xlabel('Freq [Hz]',fontsize = fontsize_axlabel)
                ax1.set_title(state + '_PSD',fontsize = fontsize_axlabel)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                #ax1.set_xlim([np.min(xf),np.max(xf)])

                if save_flag:       
                    fig1.savefig(PSD_path +os.sep+ state + '_' + str(icase) +'_comp' + format)
                plt.close(fig1)

                
            fig,ax = plt.subplots(1)
            ax.plot(time_of,states_of[:,ix],label = 'OpenFAST')
            ax.plot(time_dfsm,states_dfsm[:,ix],label = 'DFSM')
            
            ax.set_title(state,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                    
                fig.savefig(plot_path +os.sep+ state+ '_'+str(icase) +'_comp' + format)

            plt.close(fig)

        #------------------------------------------
        # Plot Outputs
        #------------------------------------------
        for iy,output in enumerate(reqd_outputs):


            if output in PSD_quantities:
                xf,FFT_of,_ = spectral.fft_wrap(time_of,outputs_of[:,iy],averaging = 'Welch',averaging_window= 'hamming')
                xf,FFT_dfsm,_ = spectral.fft_wrap(time_dfsm,outputs_dfsm[:,iy],averaging = 'Welch',averaging_window= 'hamming')

                fig1,ax1 = plt.subplots(1)
                ax1.loglog(xf,np.sqrt(FFT_of),label = 'OpenFAST')
                ax1.loglog(xf,np.sqrt(FFT_dfsm),label = 'DFSM')
                ax1.set_xlabel('Freq [Hz]',fontsize = fontsize_axlabel)
                ax1.set_title(output + '_PSD',fontsize = fontsize_axlabel)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                #ax1.set_xlim([np.min(xf),np.max(xf)])

                if save_flag:       
                    fig1.savefig(PSD_path +os.sep+ output + '_' + str(icase) +'_comp' + format)
                plt.close(fig1)

            fig,ax = plt.subplots(1)
            ax.plot(time_of,outputs_of[:,iy],label = 'OpenFAST')
            ax.plot(time_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
            
            
            ax.set_title(output,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                    
                fig.savefig(plot_path +os.sep+ output+ '_'+str(icase) + '_comp' + format)

            plt.close(fig)

        # post processing
        case_name_of = case_names_of[icase]
        case_name_dfsm = case_names_dfsm[icase]

        # compile results from DFSM
        OutData_dfsm = compile_dfsm_results(time_dfsm,states_dfsm,controls_dfsm,outputs_dfsm,reqd_states,
                                        reqd_controls,reqd_outputs,53,transition_time)

        # get output
        output_dfsm = OpenFASTOutput.from_dict(OutData_dfsm,case_name_dfsm,magnitude_channels = magnitude_channels)

        outlist_dfsm.append(output_dfsm)

        # compile OpenFAST sim results
        OutData_of = compile_dfsm_results(time_of,states_of,controls_of,outputs_of,reqd_states,
                                        reqd_controls,reqd_outputs,53,transition_time)

        # get output
        output_of = OpenFASTOutput.from_dict(OutData_of,case_name_of,magnitude_channels = magnitude_channels)

        outlist_of.append(output_of)

    
    # instantiate LA class for openfast and dfsm results
    loads_analysis_of = LoadsAnalysis(
            outputs = [],
            magnitude_channels = magnitude_channels,
            fatigue_channels = fatigue_channels
        ) 

    loads_analysis_dfsm = LoadsAnalysis(
            outputs = [],
            magnitude_channels = magnitude_channels,
            fatigue_channels = fatigue_channels
        )   
    
    # Collect outputs
    ss = {}
    et = {}
    dl = {}
    dam = {}

    for output in outlist_of:
        _name, _ss, _et, _dl, _dam = loads_analysis_of._process_output(output)
        ss[_name] = _ss
        et[_name] = _et
        dl[_name] = _dl
        dam[_name] = _dam

    summary_stats_of, extreme_table_of, DELs_of, Damage_of = loads_analysis_of.post_process(ss, et, dl, dam)

    # Collect outputs
    ss = {}
    et = {}
    dl = {}
    dam = {}

    for output in outlist_dfsm:
        _name, _ss, _et, _dl, _dam = loads_analysis_dfsm._process_output(output)
        ss[_name] = _ss
        et[_name] = _et
        dl[_name] = _dl
        dam[_name] = _dam

    summary_stats_dfsm, extreme_table_dfsm, DELs_dfsm, Damage_dfsm = loads_analysis_dfsm.post_process(ss, et, dl, dam)

    results_dict = {'sum_stats_of':summary_stats_of,
                    'sum_stats_dfsm':summary_stats_dfsm,
                    'DELs_of':DELs_of,
                    'DELs_dfsm':DELs_dfsm
                    }
    

    results_file = plot_path +os.sep+ 'DFSM_validation_results.pkl'

    with open(results_file,'wb') as handle:
        pickle.dump(results_dict,handle)
                

if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop
    
    test_inds = np.arange(0,42)

    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    #---------------------------------------------------
    # Setup MPI stuff
    #---------------------------------------------------

    if MPI:
        
        # set number of design variables and finite difference variables as 1
        n_DV = 1; n_FD = 1

        # get maximum available cores
        max_cores = MPI.COMM_WORLD.Get_size()

        # get number of cases we will be running
        n_OF_runs = len(test_inds)
        max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
        n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])

        olaf = False

        # get mapping
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(n_FD, n_OF_runs_parallel, openmp=olaf)

        rank    = MPI.COMM_WORLD.Get_rank()

        if rank < len(color_map):
            try:
                color_i = color_map[rank]
            except IndexError:
                raise ValueError('The number of finite differencing variables is {} and the correct number of cores were not allocated'.format(n_FD))
        else:
            color_i = max(color_map) + 1
        
        comm_i  = MPI.COMM_WORLD.Split(color_i, 1)
    else:
        color_i = 0
        rank = 0

    if rank == 0:
        #---------------------------------------------------
        # Load DFSM model
        #---------------------------------------------------

        # pickle with the saved DFSM model
        pkl_name = this_dir + os.sep +'dfsm_mhk.pkl'

        format = '.pdf'
        dt = 0.005;transition_time = 200

        # load dfsm model
        with open(pkl_name,'rb') as handle:
            dfsm = pickle.load(handle)
        dfsm.simulation_time = []

        # setup the interpolation of the matrices in the LPV model
        interp_type = 'nearest'
        dfsm.setup_LPV(interp_type)

        #---------------------------------------------------
        # Load OpenFAST validation results
        #---------------------------------------------------

        # datapath
        testpath = '/home/athulsun/DFSM/data/RM1_test' #<-------------------change this

        # get the path to all .outb files in the directory
        outfiles = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
        outfiles = sorted(outfiles)
        
        # required states
        reqd_states = ['PtfmPitch','PtfmHeave','GenSpeed']
        
        # required controls
        reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
        
        # required outputs
        reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt'] 
        
        # scaling parameters
        scale_args = {'state_scaling_factor': np.array([1,1,100]),
                    'control_scaling_factor': np.array([1,1,1,1]),
                    'output_scaling_factor': np.array([1,1,1,1,1,1,1])
                    }
        
        # filter parameters
        filter_args = {'state_filter_flag': [True,True,True],
                    'state_filter_type': [['filtfilt'],['filtfilt'],['filtfilt']],
                    'state_filter_tf': [[0.1],[0.1],[0.1]],
                    'control_filter_flag': [False,False,False],
                    'control_filter_tf': [0,0,0],
                    'output_filter_flag': []
                    }
        
        # name of mat file that has the linearized models
        mat_file_name = None
        
        # instantiate class
        sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                    ,add_dx2 = True,linear_model_file = mat_file_name)
        
        
        # load and process data
        sim_detail.load_openfast_sim()
        
        
        # extract data
        FAST_sim = sim_detail.FAST_sim

    #---------------------------------------------------
    # Create folders to plot the results
    #---------------------------------------------------

    # save flag
    save_flag = True

    # save path
    save_path = this_dir + os.sep + 'outputs' + os.sep +'closed_loop_validation'
    

    if rank == 0 and not os.path.isdir(save_path):
        os.makedirs(save_path)

    # separate folder to save PSD plots
    PSD_path = save_path + os.sep + 'PSD'

    if rank == 0 and not os.path.isdir(PSD_path):
        os.makedirs(PSD_path)

    # format to save the figures
    format = '.pdf'

    ode_algorithm = 'ABM4'


    if color_i == 0:
        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down

        else:

            mpi_options = {}
            mpi_options['mpi_run'] = False 
            mpi_options['mpi_comm_map_down'] = []

        # run the simulations
        run_closed_loop_simulation(dfsm,FAST_sim,dt,ode_algorithm,testpath,transition_time,save_flag,save_path,PSD_path,format,test_inds,mpi_options)

    #---------------------------------------------------
    # More MPI stuff
    #---------------------------------------------------

    if MPI and color_i < 1000000:
        sys.stdout.flush()
        if rank in comm_map_up.keys():
            subprocessor_loop(comm_map_up)
        sys.stdout.flush()

        # close signal to subprocessors
        subprocessor_stop(comm_map_down)
        sys.stdout.flush()

    if MPI and color_i < 1000000:
        MPI.COMM_WORLD.Barrier()