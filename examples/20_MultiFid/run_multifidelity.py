
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

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/below_rated_p05' + os.sep + 'openfast_runs'
    wind_dataset = OF_dir + os.sep + 'wind_dataset.pkl'

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

        # 1. DFSM file and the model detials
        dfsm_file = this_dir + os.sep + 'dfsm_fowt_1p6.pkl'

        reqd_states = ['PtfmSurge','PtfmPitch','TTDspFA','GenSpeed']
        reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
        reqd_outputs = ['TwrBsFxt','TwrBsMyt','GenPwr','YawBrTAxp','NcIMURAys','RtFldCp','RtFldCt']

        
        # 3. ROSCO yaml file
        rosco_yaml = this_dir + os.sep + 'IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'

    if color_i == 0:

        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down

        else:

            mpi_options = None
        
        mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=200,wind_dataset=wind_dataset)
        bounds = {'omega_vs' : np.array([[1, 3]]),'zeta_vs' : np.array([[0.5, 3.0]])}
        desvars = {'omega_vs' : np.array([2.5]),'zeta_vs': np.array([2.5])}
        scaling_dict = {'omega_vs':10}

        n_pts = 10

        objs = np.zeros((n_pts,2))
        opt_pts = np.zeros((n_pts,2))

        w1 = np.linspace(1,0,n_pts)
        w2 = 1-w1

        obj1 = 'TwrBsMyt_DEL'
        obj2 = 'GenSpeed_Std'

        results_folder = OF_dir + os.sep + 'multi_fid_results_br'

        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

        lf_warmstart_file = OF_dir + os.sep +'lf_ws_file_oz_25.dill'
        hf_warmstart_file = OF_dir + os.sep +'hf_ws_file_oz_25.dill'

        for i_pt in range(n_pts):


            multi_fid_dict = {'obj1':obj1,'obj2':obj2,'w1':w1[i_pt],'w2':w2[i_pt]}

            lf_warmstart_file_iter = results_folder + os.sep+'multiobj_iter_lf_'+ str(i_pt)+'.dill'
            hf_warmstart_file_iter = results_folder + os.sep+'multiobj_iter_hf_'+ str(i_pt)+'.dill'

            lf_warmstart_file_ = lf_warmstart_file
            hf_warmstart_file_ = hf_warmstart_file


            with open(lf_warmstart_file_,'rb') as handle:
                lf_res = dill.load(handle)

            with open(hf_warmstart_file_,'rb') as handle:
                hf_res = dill.load(handle)

            lf_res_iter = copy.deepcopy(lf_res)
            hf_res_iter = copy.deepcopy(hf_res)

            DV0 = np.array(hf_res['desvars'])[-1,:]
            print(DV0)
            
            for i in range(len(lf_res_iter['outputs'])):
                lf_res_iter['outputs'][i]['wt_objectives'] = lf_res_iter['outputs'][i][obj1]*w1[i_pt] + lf_res_iter['outputs'][i][obj2]*w2[i_pt]

            for i in range(len(hf_res_iter['outputs'])):
                hf_res_iter['outputs'][i]['wt_objectives'] = hf_res_iter['outputs'][i][obj1]*w1[i_pt] + hf_res_iter['outputs'][i][obj2]*w2[i_pt]

            with open(lf_warmstart_file_iter,'wb') as handle:
                dill.dump(lf_res_iter,handle)

            with open(hf_warmstart_file_iter,'wb') as handle:
                dill.dump(hf_res_iter,handle)

            model_low = LFTurbine(desvars,  mf_turb, scaling_dict = scaling_dict,multi_fid_dict=multi_fid_dict,warmstart_file = lf_warmstart_file_iter)
            model_high = HFTurbine(desvars, mf_turb, scaling_dict = scaling_dict,multi_fid_dict=multi_fid_dict, warmstart_file = hf_warmstart_file_iter)

            np.random.seed(123)

            trust_region = SimpleTrustRegion(
                model_low,
                model_high,
                bounds,
                disp=2,
                trust_radius=0.5,
                num_initial_points=0,
                radius_tol = 1e-3,
                optimization_log = True,
                log_filename = results_folder + os.sep +'MO_DEL_STD_'+str(i_pt)+'.txt'
            )

            trust_region.add_objective("wt_objectives", scaler = 1e-0)
            trust_region.design_vectors = np.array(hf_res_iter['desvars'])
            #trust_region.add_constraint("GenSpeed_Max", upper=1.2)
            trust_region.set_initial_point(DV0)


            t1 = timer.time()
            trust_region.optimize(plot=False, num_basinhop_iterations=2,num_iterations = 40)
            t2 = timer.time()

            opt_pts[i_pt,:] = trust_region.design_vectors[-1,:]
            objs[i_pt,0] = trust_region.model_high.run(opt_pts[i_pt,:])[obj1]
            objs[i_pt,1] = trust_region.model_high.run(opt_pts[i_pt,:])[obj2]

            
        fig,ax = plt.subplots(1)

        ax.plot(objs[:,0],objs[:,1],'.',markersize = 8)
        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)

        fig.savefig(results_folder + os.sep +'DELvsSTD.png')
        print(objs)
        print(opt_pts)





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

    