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

# DFSM modules

from weis.dfsm.ode_algorithms import RK4,ABM4
from scipy.interpolate import interp1d

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension,calculate_MSE

from wisdem.commonse.mpi_tools import MPI

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 12

# parameters
bins = 10
slope = 4


def evaluate_multi(case_data):

    # initialize controller interface
    case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])

    # run simulation
    t1 = timer.time()
    output = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
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
        t,x,u,y = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
        
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




def run_closed_loop_simulation(dfsm,FAST_sim,dt,transition_time,save_flag,plot_path,PSD_path,format,test_inds,mpi_options):

    # path to DISCON library
    lib_name = '/home/athulsun/anaconda3/envs/weis-dev/lib/libdiscon.so'
     
    # Write parameter input file
    param_filename = '/home/athulsun/WEIS-AKS-commit/examples/19_DFSM/IEA_15_MW/IEA_w_TMD_DISCON.IN'
   

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
        t0 = 0;tf = 600
        time_of = np.linspace(t0,tf,nt)

        T_of.append(time_of)
        
        w_fun = CubicSpline(time_of, wind_speed)
        bp0 = test_data['controls'][0,2]
        gt0 = test_data['controls'][0,1]
        print(bp0);print(gt0)
        
        if reqd_controls[-1] == 'Wave1Elev':
            wave_elev = test_data['controls'][:,3]
            wave_fun = CubicSpline(time_of,wave_elev)
        else:
            wave_fun = None
        
        # time span
        tspan = [t0,tf]
        
        x0 = test_data['states'][0,:]
        
        num_blade = int(3)

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
                    'w_fun':w_fun,
                    'gen_speed_scaling':1,
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

        case_data_all.append(case_data)

    if mpi_options['mpi_run']:

        # evaluate the closed loop simulations in parallel using MPI

        outputs = run_mpi(case_data_all,mpi_options)

    else:

        # evaluate the closed loop simulations serially
        
        outputs = run_serial(case_data_all)

    MSE_dict = {}
    mean_dict = {}
    std_dict = {}
    max_dict = {}
    min_dict = {}
    PSD_dict = {}

    PSD_quantities = ['GenSpeed','TwrBsMyt','PtfmPitch','BldPitch1','GenTq','Wave1Elev']

    BldPitch_array = []
    currspeed_array = []
    GenPwr_array = []
    TwrBsMyt_array = []
    PtfmPitch_array = []
    DEL_array = []
    #-----------------------------------------------
    # Plot results
    #-----------------------------------------------

    
    elapsed = tf - transition_time
    tf = tf - transition_time
    tspan = [t0,tf]

    for icase,sim_result in enumerate(outputs):

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
            MSE_dict[control+ '_'+str(icase)] = calculate_MSE(controls_of[:,iu],controls_dfsm[:,iu],scaled = False)

            mean_dict[control+'_of_'+ str(icase)] = np.mean(controls_of[:,iu])
            mean_dict[control+'_dfsm_'+ str(icase)] = np.mean(controls_dfsm[:,iu])

            max_dict[control+'_of_'+ str(icase)] = np.max(controls_of[:,iu])
            max_dict[control+'_dfsm_'+ str(icase)] = np.max(controls_dfsm[:,iu])

            min_dict[control+'_of_'+ str(icase)] = np.min(controls_of[:,iu])
            min_dict[control+'_dfsm_'+ str(icase)] = np.min(controls_dfsm[:,iu])

            std_dict[control+'_of_'+ str(icase)] = np.std(controls_of[:,iu])
            std_dict[control+'_dfsm_'+ str(icase)] = np.std(controls_dfsm[:,iu])

            if control == 'BldPitch1':
                bp_dict = {'OpenFAST':controls_of[:,iu],'DFSM':controls_dfsm[:,iu]}
                BldPitch_array.append(bp_dict)

            if control == 'RtVAvgxh':
                dict = {'RtVAvgxh':controls_of[:,iu]}
                currspeed_array.append(dict)

            if control in PSD_quantities:
                xf,FFT_of,_ = spectral.fft_wrap(time_of,controls_of[:,iu],averaging = 'Welch',averaging_window= 'hamming')
                xf,FFT_dfsm,_ = spectral.fft_wrap(time_dfsm,controls_dfsm[:,iu],averaging = 'Welch',averaging_window= 'hamming')

                fig1,ax1 = plt.subplots(1)
                ax1.loglog(xf,np.sqrt(FFT_of),label = 'OpenFAST')
                ax1.loglog(xf,np.sqrt(FFT_dfsm),label = 'DFSM')
                ax1.set_xlabel('Freq [Hz]',fontsize = fontsize_axlabel)
                ax1.set_title(control + '_PSD',fontsize = fontsize_axlabel)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                ax1.set_xlim([np.min(xf),np.max(xf)])

                if save_flag:       
                    fig1.savefig(PSD_path +os.sep+ control + '_' + str(icase) +'_comp' + format)
                plt.close(fig1)

                PSD_dict['v_avg' + str(icase)] = np.mean(controls_of[:,0])
                PSD_dict['xf' + str(icase)] = xf
                PSD_dict[control + '_FFT_of_'+str(icase)] = np.sqrt(FFT_of)
                PSD_dict[control + '_FFT_dfsm_'+str(icase)] = np.sqrt(FFT_dfsm)

        
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

            MSE_dict[state+ '_' +str(icase)] = calculate_MSE(states_of[:,ix],states_dfsm[:,ix],scaled = False)

            mean_dict[state+'_of_'+ str(icase)] = np.mean(states_of[:,ix])
            mean_dict[state+'_dfsm_'+ str(icase)] = np.mean(states_dfsm[:,ix])

            max_dict[state+'_of_'+ str(icase)] = np.max(states_of[:,ix])
            max_dict[state+'_dfsm_'+ str(icase)] = np.max(states_dfsm[:,ix])

            min_dict[state+'_of_'+ str(icase)] = np.min(states_of[:,ix])
            min_dict[state+'_dfsm_'+ str(icase)] = np.min(states_dfsm[:,ix])

            std_dict[state+'_of_'+ str(icase)] = np.std(states_of[:,ix])
            std_dict[state+'_dfsm_'+ str(icase)] = np.std(states_dfsm[:,ix])

            if state == 'PtfmPitch':
                dict = {'OpenFAST':states_of[:,ix],'DFSM':states_dfsm[:,ix]}
                PtfmPitch_array.append(dict)

            if state in PSD_quantities:
                xf,FFT_of,_ = spectral.fft_wrap(time_of,states_of[:,ix],averaging = 'Welch',averaging_window= 'hamming')
                xf,FFT_dfsm,_ = spectral.fft_wrap(time_dfsm,states_dfsm[:,ix],averaging = 'Welch',averaging_window= 'hamming')

                fig1,ax1 = plt.subplots(1)
                ax1.loglog(xf,np.sqrt(FFT_of),label = 'OpenFAST')
                ax1.loglog(xf,np.sqrt(FFT_dfsm),label = 'DFSM')
                ax1.set_xlabel('Freq [Hz]',fontsize = fontsize_axlabel)
                ax1.set_title(state + '_PSD',fontsize = fontsize_axlabel)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                ax1.set_xlim([np.min(xf),np.max(xf)])

                if save_flag:       
                    fig1.savefig(PSD_path +os.sep+ state + '_' + str(icase) +'_comp' + format)
                plt.close(fig1)

                PSD_dict['v_avg' + str(icase)] = np.mean(controls_of[:,0])
                PSD_dict['xf' + str(icase)] = xf
                PSD_dict[state + '_FFT_of_'+str(icase)] = np.sqrt(FFT_of)
                PSD_dict[state + '_FFT_dfsm_'+str(icase)] = np.sqrt(FFT_dfsm)
                
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

            MSE_dict[output+ '_' +str(icase)] = calculate_MSE(outputs_of[:,iy],outputs_dfsm[:,iy],scaled = False)

            mean_dict[output+'_of_'+ str(icase)] = np.mean(outputs_of[:,iy])
            mean_dict[output+'_dfsm_'+ str(icase)] = np.mean(outputs_dfsm[:,iy])

            max_dict[output+'_of_'+ str(icase)] = np.max(outputs_of[:,iy])
            max_dict[output+'_dfsm_'+ str(icase)] = np.max(outputs_dfsm[:,iy])

            min_dict[output+'_of_'+ str(icase)] = np.min(outputs_of[:,iy])
            min_dict[output+'_dfsm_'+ str(icase)] = np.min(outputs_dfsm[:,iy])

            std_dict[output+'_of_'+ str(icase)] = np.std(outputs_of[:,iy])
            std_dict[output+'_dfsm_'+ str(icase)] = np.std(outputs_dfsm[:,iy])

            if output == 'TwrBsMyt':
                dict = {'OpenFAST':outputs_of[:,iy],'DFSM':outputs_dfsm[:,iy]}
                TwrBsMyt_array.append(dict)

            if output == 'TwrBsMyt':
                TwrBsMyt_of = outputs_of[:,iy]
                TwrBsMyt_dfsm = outputs_dfsm[:,iy]

                F_of, Fmean_of = fatpack.find_rainflow_ranges(TwrBsMyt_of, return_means=True)
                Nrf_of, Frf_of = fatpack.find_range_count(F_of, bins)
                DELs_ = Frf_of ** slope * Nrf_of / elapsed

                dict = {}
                dict['OpenFAST'] = DELs_.sum() ** (1.0 / slope)

                F_dfsm, Fmean_dfsm = fatpack.find_rainflow_ranges(TwrBsMyt_dfsm, return_means=True)
                Nrf_dfsm, Frf_dfsm = fatpack.find_range_count(F_dfsm, bins)
                DELs_ = Frf_dfsm ** slope * Nrf_dfsm / elapsed
                dict['DFSM'] = DELs_.sum() ** (1.0 / slope)

                DEL_array.append(dict)

            if output == 'GenPwr':
                dict = {'OpenFAST':outputs_of[:,iy],'DFSM':outputs_dfsm[:,iy]}
                GenPwr_array.append(dict)

            if output in PSD_quantities:
                xf,FFT_of,_ = spectral.fft_wrap(time_of,outputs_of[:,iy],averaging = 'Welch',averaging_window= 'hamming')
                xf,FFT_dfsm,_ = spectral.fft_wrap(time_dfsm,outputs_dfsm[:,iy],averaging = 'Welch',averaging_window= 'hamming')

                fig1,ax1 = plt.subplots(1)
                ax1.loglog(xf,np.sqrt(FFT_of),label = 'OpenFAST')
                ax1.loglog(xf,np.sqrt(FFT_dfsm),label = 'DFSM')
                ax1.set_xlabel('Freq [Hz]',fontsize = fontsize_axlabel)
                ax1.set_title(output + '_PSD',fontsize = fontsize_axlabel)
                ax1.legend(ncol = 2,fontsize = fontsize_legend)
                ax1.set_xlim([np.min(xf),np.max(xf)])

                if save_flag:       
                    fig1.savefig(PSD_path +os.sep+ output + '_' + str(icase) +'_comp' + format)
                plt.close(fig1)

                PSD_dict['v_avg' + str(icase)] = np.mean(controls_of[:,0])
                PSD_dict['xf' + str(icase)] = xf
                PSD_dict[output + '_FFT_of_'+str(icase)] = np.sqrt(FFT_of)
                PSD_dict[output + '_FFT_dfsm_'+str(icase)] = np.sqrt(FFT_dfsm)

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

    results_dict = {'MSE_dict':MSE_dict,
                    'mean_dict':mean_dict,
                    'std_dict':std_dict,
                    'PSD_dict':PSD_dict,
                    'max_dict':max_dict,
                    'min_dict':min_dict,
                    'Time':time_of,
                    'BldPitch_array':BldPitch_array,
                    'GenPwr_array':GenPwr_array,
                    'TwrBsMyt_array':TwrBsMyt_array,
                    'PtfmPitch_array':PtfmPitch_array,
                    'DEL_array':DEL_array}
    

    results_file = plot_path +os.sep+ 'DFSM_validation_results.pkl'
    with open(results_file,'wb') as handle:
        pickle.dump(results_dict,handle)
                

if __name__ == '__main__':

    if MPI:
        from wisdem.commonse.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop
    
    test_inds = np.array([15,16,17,18,19,
                          25,26,27,28,29,
                          35,36,37,38,39,
                          45,46,47,48,49,
                          55,56,57,58,59,
                          65,66,67,68,69,
                          75,76,77,78,79,
                          85,86,87,88,89,
                          95,96,97,98,99,
                          105,106,107,108,109])


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

        # path to this directory
        this_dir = os.path.dirname(os.path.abspath(__file__))

        # pickle with the saved DFSM model
        pkl_name = this_dir + os.sep +'dfsm_1p6_python.pkl'

        format = '.pdf'
        dt = 0.01;transition_time = 50

        # load dfsm model
        with open(pkl_name,'rb') as handle:
            dfsm = pickle.load(handle)
        dfsm.simulation_time = []

        # setup the interpolation of the matrices in the LPV model
        # interp_type = 'nearest'
        # dfsm.setup_LPV(interp_type)

        #---------------------------------------------------
        # Load OpenFAST validation results
        #---------------------------------------------------

        # datapath
        testpath = '/home/athulsun/DFSM/data' + os.sep + 'FOWT_1p6'

        # get the path to all .outb files in the directory
        outfiles = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
        outfiles = sorted(outfiles)
        
        # required states
        reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
        
        # required controls
        reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
        
        # required outputs
        reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt'] 
        
        # scaling parameters
        scale_args = {'state_scaling_factor': np.array([1,1,1]),
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
    save_path = 'validation_results_1p6'
    

    if rank == 0 and not os.path.isdir(save_path):
        os.makedirs(save_path)

    # separate folder to save PSD plots
    PSD_path = save_path + os.sep + 'PSD'

    if rank == 0 and not os.path.isdir(PSD_path):
        os.makedirs(PSD_path)

    # format to save the figures
    format = '.pdf'


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
        run_closed_loop_simulation(dfsm,FAST_sim,dt,transition_time,save_flag,save_path,PSD_path,format,test_inds,mpi_options)

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