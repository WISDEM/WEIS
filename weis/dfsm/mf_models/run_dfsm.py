import numpy as np
import time as timer
from rosco.toolbox import control_interface as ROSCO_ci
from weis.dfsm.ode_algorithms import RK4,ABM4
from weis.dfsm.dfsm_utilities import compile_dfsm_results
from pCrunch import Crunch,FatigueParams, AeroelasticOutput


def evaluate_multi(case_data):

    # initialize controller interface
    case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])

    # run simulation
    t1 = timer.time()
    if case_data['ode_method'] == 'RK4':

        output = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

    elif case_data['ode_method'] == 'ABM4':

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
        t1 = timer.time()
        if case_data['ode_method'] == 'RK4':
            t,x,u,y = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

        elif case_data['ode_method'] == 'ABM4':
            t,x,u,y = ABM4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
        t2 = timer.time()
        print(t2-t1)
        
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

    from openmdao.utils.mpi import MPI

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


def run_dfsm(case_data_all,reqd_states,reqd_controls,reqd_outputs,mpi_options,GB_ratio = 1, TStart = 0):

    if mpi_options['mpi_run']:
        # evaluate the closed loop simulations in parallel using MPI
        sim_outputs = run_mpi(case_data_all,mpi_options)

    else:
        # evaluate the closed loop simulations serially
        sim_outputs = run_serial(case_data_all)

    chan_time_list = []
    ae_output_list = []
    fatigue_channels = {'TwrBsMyt': FatigueParams(slope=4),}

    for i_case,sim_result in enumerate(sim_outputs):

        # extract results
        T_dfsm = sim_result['T_dfsm']

        states_dfsm = sim_result['states_dfsm']

        controls_dfsm = sim_result['controls_dfsm']

        outputs_dfsm = sim_result['outputs_dfsm']

        # compile results from DFSM
        OutData = compile_dfsm_results(T_dfsm,states_dfsm,controls_dfsm,outputs_dfsm,reqd_states,
                                        reqd_controls,reqd_outputs,GB_ratio,TStart)
        
        chan_time_list.append(OutData)

        # get output
        ae_output = AeroelasticOutput(OutData, dlc = 'dfsm_'+str(i_case),  fatigue_channels = fatigue_channels )

        ae_output_list.append(ae_output)

    cruncher = Crunch(outputs = [],lean = True)

    for output in ae_output_list:
        cruncher.add_output(output)

    cruncher.outputs = ae_output_list

    return cruncher,ae_output_list,chan_time_list







