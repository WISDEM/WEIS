
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from models.prod_functions import LFTurbine,HFTurbine
from models.mf_controls import MF_Turbine,compute_outputs,valid_extension
from weis.multifidelity.methods.trust_region import SimpleTrustRegion
from weis.glue_code.mpi_tools import MPI
import pickle,dill,copy
import time as timer

if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 1. DFSM file and the model detials
    dfsm_file = os.path.join(this_dir,'..', "09_DFSM", "dfsm_models", "dfsm_iea15_volturnus.pkl")

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = '/Users/dzalkind/Tools/WEIS-CSU/examples/09_DFSM/outputs/IEA-15/2_for_mf/openfast_runs'
    wind_dataset = os.path.join(OF_dir, 'wind_dataset.pkl')

    # 3. ROSCO yaml file
    rosco_yaml = os.path.join(this_dir, '..', "00_setup", "OpenFAST_models", "IEA-15-240-RWT", "IEA-15-240-RWT-UMaineSemi", "IEA-15-240-RWT-UMaineSemi_ROSCO.yaml")

    # 4. Startup transients
    transition_time = 0

    # 5. Set DVs and bounds
    bounds = {'omega_pc' : np.array([[1, 3]]),'zeta_pc' : np.array([[0.5, 3.0]])}
    desvars = {'omega_pc' : np.array([2.5]),'zeta_pc': np.array([2.5])}
    scaling_dict = {'omega_pc':10}

    # 6. Solver settings
    num_basinhop_iterations=0
    num_iterations = 2

    fst_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*.fst')]
    n_OF_runs = len(fst_files)

    # flag to indicate multi objective study
    MO_flag = False


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
        
        mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=transition_time,wind_dataset=wind_dataset)
        


        if MO_flag:
            n_pts = 10

            objs = np.zeros((n_pts,2))
            opt_pts = np.zeros((n_pts,2))

            w1 = np.linspace(1,0,n_pts)
            w2 = 1-w1

            obj1 = 'TwrBsMyt_DEL'
            obj2 = 'GenSpeed_Std'

        else:
            n_pts = 1

        # create folder if it does not exists
        results_folder = OF_dir + os.sep + 'multi_fid_results'

        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

        # Go through and run multifidelity studies.
        # If MO_flag is enabled, different multifidelity studies are run for different values of the weights
        # Else, just a single multi-fidelity study is run
        for i_pt in range(n_pts):

            if MO_flag:
                multi_fid_dict = {'obj1':obj1,'obj2':obj2,'w1':w1[i_pt],'w2':w2[i_pt]}
            else:
                multi_fid_dict = None

            model_low = LFTurbine(desvars,  mf_turb, scaling_dict = scaling_dict,multi_fid_dict=multi_fid_dict)
            model_high = HFTurbine(desvars, mf_turb, scaling_dict = scaling_dict,multi_fid_dict=multi_fid_dict)

            np.random.seed(123)

            trust_region = SimpleTrustRegion(
                model_low,
                model_high,
                bounds,
                disp=2,
                trust_radius=0.5,
                num_initial_points=2,
                radius_tol = 1e-3,
                optimization_log = True,
                log_filename = results_folder + os.sep +'MO_DEL_STD_'+str(i_pt)+'.txt'
            )

            if MO_flag:

                trust_region.add_objective("wt_objectives", scaler = 1e-0)
            else:
                
                trust_region.add_objective("TwrBsMyt_DEL", scaler = 1)
                trust_region.add_constraint("GenSpeed_Max", upper=1.2)
            

            # run multi-fidelity study
            t1 = timer.time()
            trust_region.optimize(
                plot=False, 
                num_basinhop_iterations=num_basinhop_iterations,
                num_iterations=num_iterations
            )
            t2 = timer.time()

            if MO_flag:
                opt_pts[i_pt,:] = trust_region.design_vectors[-1,:]
                objs[i_pt,0] = trust_region.model_high.run(opt_pts[i_pt,:])[obj1]
                objs[i_pt,1] = trust_region.model_high.run(opt_pts[i_pt,:])[obj2]

        if MO_flag:

            # plot pareto front
            fig,ax = plt.subplots(1)

            ax.plot(objs[:,0],objs[:,1],'.',markersize = 8)
            ax.set_xlabel(obj1)
            ax.set_ylabel(obj2)

            fig.savefig(results_folder + os.sep +'DELvsSTD.png')

            # save results
            results_dict = {'objectives':objs,'opt_pts':opt_pts,'n_pts':n_pts,'obj1':obj1,'obj2':obj2}

            with open(results_folder + os.sep + 'MO_results.pkl','wb') as handle:
                pickle.dump(results_dict,handle)

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

    