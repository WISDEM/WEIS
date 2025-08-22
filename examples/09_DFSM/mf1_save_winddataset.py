import numpy as np
import os,sys
import matplotlib.pyplot as plt
from weis.dfsm.mf_models.mf_controls import MF_Turbine,compute_outputs,valid_extension
from weis.glue_code.mpi_tools import MPI
import pickle

if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 1. DFSM file and the model detials
    dfsm_file = os.path.join(this_dir, "dfsm_models", "dfsm_iea15_volturnus.pkl")

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = os.path.join(this_dir, 'outputs' ,'IEA-15', '0_dfsm_setup', 'openfast_runs')

    # 3. ROSCO yaml file
    rosco_yaml = os.path.join(this_dir, '..', "00_setup", "OpenFAST_models", "IEA-15-240-RWT", "IEA-15-240-RWT-UMaineSemi", "IEA-15-240-RWT-UMaineSemi_ROSCO.yaml")

    fst_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*.fst')]
    n_OF_runs = len(fst_files)

    if MPI:
        
        # set number of design variables and finite difference variables as 1
        n_DV = 1; n_FD = 1

        # get maximum available cores
        max_cores = MPI.COMM_WORLD.Get_size()

        # get number of cases we will be running
        max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
        n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])

        olaf = False

        # get mapping
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(n_FD, n_OF_runs_parallel)

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

        

        with open(dfsm_file,'rb') as handle:
            dfsm = pickle.load(handle)

        reqd_states = dfsm.reqd_states
        reqd_controls = dfsm.reqd_controls
        reqd_outputs = dfsm.reqd_outputs

        

    if color_i == 0:

        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down

        else:

            mpi_options = None
        
        mf_controls = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=00)

        fig_fol = OF_dir + os.sep + 'plot_comp'
        if not os.path.exists(fig_fol):
            os.mkdir(fig_fol)

        cruncher_dfsm,ae_output_list_dfsm,chan_time_list_dfsm = mf_controls.run_dfsm()

        outputs_dfsm = compute_outputs(cruncher_dfsm)
        print(outputs_dfsm)

        cruncher_of,ae_output_list_of,chan_time_list_of = mf_controls.run_openfast()
        outputs_of = compute_outputs(cruncher_of)
        print(outputs_of)

        i_fig = 0

        nt = len(chan_time_list_of[0]['Time'])

        wind_dataset = np.zeros((nt,n_OF_runs+1))
        wind_dataset[:,0] = chan_time_list_of[0]['Time']
        
        for ct_of,ct_dfsm in zip(chan_time_list_of,chan_time_list_dfsm):

            channels = mf_controls.channels

            wind_dataset[:,i_fig+1] = ct_of['RtVAvgxh']

            for chan in channels:

                fig,ax = plt.subplots(1)
                fig.subplots_adjust(hspace = 0.65)

                ax.plot(ct_of['Time'],ct_of[chan],label = 'OpenFAST')
                ax.plot(ct_dfsm['Time'],ct_dfsm[chan],label = 'DFSM')
                ax.legend(ncol = 2)

                fig.savefig(fig_fol + os.sep +chan +'_comp'+str(i_fig)+'.pdf')
                plt.close(fig)
            i_fig+=1

        with open(OF_dir + os.sep +'wind_dataset.pkl','wb') as handle:
            pickle.dump(wind_dataset,handle)

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

    