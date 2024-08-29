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
from dfsm.dfsm_utilities import valid_extension,calculate_time, calculate_MSE, extrapolate_controls
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
from dfsm.evaluate_dfsm import evaluate_dfsm
from scipy.integrate import solve_ivp

from dfsm.dfsm_rosco_simulation import run_sim_ROSCO
from dfsm.ode_algorithms import RK4

from wisdem.commonse.mpi_tools import MPI


def evaluate_multi(case_data):

    # initialize controller interface
    case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])

    # run simulation
    output = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

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




def run_closed_loop_simulation(data_path,test_path,plot_path,test_inds,mpi_options):



    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # path to DISCON library
    lib_name = os.path.join('/home/athulsun/anaconda3/envs/weis-env/lib','libdiscon.so')
    
    
    # Write parameter input file
    param_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_DISCON.IN')
   
    # datapath
    region = 'LPV'
    datapath =  '/home/athulsun/DFSM/data' + os.sep + data_path
    testpath = '/home/athulsun/DFSM/data' + os.sep + test_path 


    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)

    outfiles_test = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)
    
    # required states
    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    
    # required controls
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    
    # required outputs
    reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr'] 
    
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
    
    # name of mat file that has the linearized models
    mat_file_name = this_dir + os.sep + 'FOWT_1p6_LPV.mat'
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    sim_detail_test = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    
    # load and process data
    sim_detail.load_openfast_sim()
    sim_detail_test.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail_test.FAST_sim
    
    
    # split of training-testing data
    n_samples = 1
    
    
    # construct surrogate model
    dfsm_model = DFSM(sim_detail,n_samples = n_samples,L_type = 'LPV',N_type = None, train_split = 0.5)
    dfsm_model.construct_surrogate()
    dfsm_model.simulation_time = []

    T_of = []
    X_OF_list = []
    U_OF_list = []
    Y_OF_list = []

    case_data_all = []

    for idx,ind in enumerate(test_inds):

        case_data = {}
        case_data['case'] = idx
            
        test_data = FAST_sim[ind]
        X_OF_list.append(test_data['states'])
        U_OF_list.append(test_data['controls'])
        Y_OF_list.append(test_data['outputs'])
        
        wind_speed = test_data['controls'][:,0] #
        nt = len(wind_speed)
        t0 = 0;tf = 600
        dt = 0.0; 
        time_of = np.linspace(t0,tf,nt)

        T_of.append(time_of)
        
        w_fun = CubicSpline(time_of, wind_speed)
        bp0 = 16
        gt0 = 19000 #8.3033588
        
        if reqd_controls[-1] == 'Wave1Elev':
            wave_elev = test_data['controls'][:,3]
            wave_fun = CubicSpline(time_of,wave_elev)
            
        else:
            wave_fun = None
        
        # time span
        tspan = [t0,tf]
        
        x0 = test_data['states'][0,:]
        

        dt = 0.01
        num_blade = int(3)

        args = {'DT': dt, 'num_blade': num_blade}

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
                    'w_fun':w_fun,
                    'gen_speed_scaling':scale_args['state_scaling_factor'][-1],
                    'lib_name':lib_name,
                    'param_filename':param_filename,
                    'wave_fun':wave_fun,
                    'args':args,
                    'ny': len(reqd_outputs)
                    }
        
        case_data['param'] = param 
        case_data['dt'] = dt 
        case_data['x0'] = x0 
        case_data['tspan'] = tspan 
        case_data['dfsm'] = dfsm_model

        case_data_all.append(case_data)

    if mpi_options['mpi_run']:

        # evaluate the closed loop simulations in parallel using MPI

        outputs = run_mpi(case_data_all,mpi_options)

    else:

        # evaluate the closed loop simulations serially
        
        outputs = run_serial(case_data_all)

    #-----------------------------------------------
    # Plot results
    #-----------------------------------------------

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 18
    fontsize_tick = 12

    save_flag = True

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

        #------------------------------------------------------
        # Plot Controls and Inputs
        #------------------------------------------------------

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
                fig.savefig(plot_path +os.sep+ control + '_' + str(icase) +'_comp.pdf')

            plt.close(fig)

            

        #------------------------------------------------------
        # Plot States
        #------------------------------------------------------
        for ix,state in enumerate(reqd_states):
                
            fig,ax = plt.subplots(1)
            ax.plot(time_of,states_of[:,ix],label = 'OpenFAST')
            ax.plot(time_dfsm,states_dfsm[:,ix],label = 'DFSM')
            
            ax.set_title(state,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                    
                fig.savefig(plot_path +os.sep+ state+ '_'+str(icase) +'_comp.pdf')

            plt.close(fig)

        #------------------------------------------
        # Plot Outputs
        #------------------------------------------
        for iy,output in enumerate(reqd_outputs):

            fig,ax = plt.subplots(1)
            ax.plot(time_of,outputs_of[:,iy],label = 'OpenFAST')
            ax.plot(time_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
            
            
            ax.set_title(output,fontsize = fontsize_axlabel)
            ax.set_xlim(tspan)
            ax.tick_params(labelsize=fontsize_tick)
            ax.legend(ncol = 2,fontsize = fontsize_legend)
            ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
            
            if save_flag:
                    
                fig.savefig(plot_path +os.sep+ output+ '_'+str(icase) + '_comp.pdf')

            plt.close(fig)
                

if __name__ == '__main__':

    if MPI:
        from wisdem.commonse.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop
    
    test_inds = [12,13,14,15,16,17,18]

    if MPI:
    
        n_DV = 1
        max_cores = MPI.COMM_WORLD.Get_size()

        n_FD = n_DV

        n_OF_runs = len(test_inds)
        max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
        n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])

        olaf = False

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

    

    data_path = 'FOWT_1p6'
    test_path = '1p6_test'

    if MPI:
        save_path = 'plots_ROSCO_parallel'
    else:
        save_path = 'plots_ROSCO_serial'

    if rank == 0 and not os.path.isdir(save_path):
        os.makedirs(save_path)


    if color_i == 0:

    

        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down
  

        else:

            mpi_options = {}
            mpi_options['mpi_run'] = False 
            mpi_options['mpi_comm_map_down'] = []


        run_closed_loop_simulation(data_path,test_path,save_path,test_inds,mpi_options)


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

    
    







    

        



