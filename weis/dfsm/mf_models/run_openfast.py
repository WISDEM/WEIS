import os
import numpy as np
import shutil

from weis.aeroelasticse.FAST_wrapper import FAST_wrapper
from weis.dfsm.dfsm_utilities import compile_dfsm_results
from pCrunch import Crunch,FatigueParams, AeroelasticOutput,read

of_path = os.path.realpath( shutil.which('openfast') )

fail_value = 9999
allow_fails = True
fatigue_channels = {'TwrBsMyt': FatigueParams(slope=4),}

def run_wrapper(fst_file,overwrite_flag):

    wrapper = FAST_wrapper()

    wrapper.FAST_exe = of_path

    FAST_directory,FAST_InputFile = os.path.split(fst_file)

    wrapper.FAST_directory = FAST_directory
    wrapper.FAST_InputFile = FAST_InputFile

    FAST_Output = os.path.join(FAST_directory, FAST_InputFile[:-3]+'outb')

    output_flag = (os.path.exists(FAST_Output))

    if overwrite_flag or (not overwrite_flag and not output_flag):
        failed = wrapper.execute()
        if failed:
            print('OpenFAST Failed! Please check the run logs.')
            if allow_fails:
                print(f'OpenFAST failures are allowed. All outputs set to {fail_value}')
            else:
                raise Exception('OpenFAST Failed! Please check the run logs.')

    else:

        failed = False
        print('OpenFAST not executed: Output file "%s" already exists. To overwrite this output file, set "overwrite_outfiles = True".'%FAST_Output)


    output = read(FAST_Output,fatigue_channels = fatigue_channels)

    return output


def run_serial(fst_files,overwrite_flag):

    output_list = []
    # run openfast simulations in parallel
    for fst_file in fst_files:
        output = run_wrapper(fst_file,overwrite_flag)

        output_list.append(output)

    return output_list

def evaluate_multi(case_data):
    print(case_data['fst_file'])

    output = run_wrapper(case_data['fst_file'],case_data['overwrite_flag'])

    return output



def run_mpi(case_data_all,mpi_options):

    from openmdao.utils.mpi import MPI

    # mpi comm management
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sub_ranks = mpi_options['mpi_comm_map_down'][rank]

    size = len(sub_ranks)

    N_cases = len(case_data_all)
    N_loops = int(np.ceil(float(N_cases)/float(size)))

    output_list = []

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
            output_list.append(data_out)

    return output_list






def run_openfast(fst_files,reqd_states,reqd_controls,reqd_outputs,mpi_options,GB_ratio = 1, TStart = 0,overwrite_flag = False):

    if mpi_options['mpi_run']:
        # evaluate the closed loop simulations in parallel using MPI

        case_data_all = []
        for fst_file in fst_files:
            case_data = {}
            case_data['fst_file'] = fst_file
            case_data['overwrite_flag'] = overwrite_flag

            case_data_all.append(case_data)

        sim_outputs = run_mpi(case_data_all,mpi_options)

    else:

        # evaluate the closed loop simulations serially
        sim_outputs = run_serial(fst_files,overwrite_flag)

    chan_time_list = []
    ae_output_list = []
    fatigue_channels = {
                    'TwrBsMyt': FatigueParams(slope=4),
                                        }

    for i_case,sim_result in enumerate(sim_outputs):

        # extract results
        Time = sim_result['Time']

        nt = len(Time)

        states = np.zeros((nt,len(reqd_states)))
        controls = np.zeros((nt,len(reqd_controls)))
        outputs = np.zeros((nt,len(reqd_outputs)))

        for ix,state in enumerate(reqd_states):
            states[:,ix] = sim_result[state]

        for iu,control in enumerate(reqd_controls):
            controls[:,iu] = sim_result[control]

        for iy,output in enumerate(reqd_outputs):
            outputs[:,iy] = sim_result[output]

    

        # compile results from DFSM
        OutData = compile_dfsm_results(Time,states,controls,outputs,reqd_states,
                                        reqd_controls,reqd_outputs,GB_ratio,TStart)
        
        chan_time_list.append(OutData)

        # get output
        ae_output = AeroelasticOutput(OutData, dlc = 'openfast_'+str(i_case),  fatigue_channels = fatigue_channels )

        ae_output_list.append(ae_output)

    cruncher = Crunch(outputs = [],lean = True)

    for output in ae_output_list:
        cruncher.add_output(output)

    cruncher.outputs = ae_output_list


    return cruncher,ae_output_list,chan_time_list
