import numpy as np
from models.prod_functions import LFTurbine,HFTurbine
from models.mf_controls import MF_Turbine,valid_extension
from time import time
from os import path
import openmdao.api as om
import os,sys
from weis.glue_code.mpi_tools import MPI
from wisdem.optimization_drivers.nlopt_driver import NLoptDriver


if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    bounds = np.array([[1.0, 3.0],[0.1,3.0]])
    desvars = {'omega_pc' : np.array([2.5]),'zeta_pc':np.array([2.5])}

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/nearrated_5' + os.sep + 'openfast_runs'
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
        

        class Model(om.ExplicitComponent):
            def initialize(self):
                self.options.declare('desvars')
                
            def setup(self):
                desvars = self.options["desvars"]
                for key in desvars:
                    self.add_input(key, val=desvars[key])
                
                self.add_output('TwrBsMyt_DEL', val=0.)
                self.add_output('GenSpeed_Max', val=0.)
                
                
                mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=200,wind_dataset=wind_dataset)
                scaling_dict = {'omega_pc':10}
                self.model = LFTurbine(desvars, mf_turb, scaling_dict = scaling_dict)

            def compute(self, inputs, outputs):
                # desvars = self.options["desvars"]
                model_outputs = self.model.compute(inputs)
                outputs['TwrBsMyt_DEL'] = model_outputs['TwrBsMyt_DEL']
                outputs['GenSpeed_Max'] = model_outputs['GenSpeed_Max']
            
            
        p = om.Problem(model=om.Group(num_par_fd = 1),comm = comm_i,reports = False)
        model = p.model
        # model.approx_totals(method='fd', step=1e-3, form='central')
        comp = model.add_subsystem('Model', Model(desvars=desvars), promotes=['*'])

        for key in desvars:
            model.set_input_defaults(key, val=desvars[key])
            
        s = time()
        
        p.driver = NLoptDriver()
        p.driver.options['optimizer'] = "LN_COBYLA"
        p.driver.options['tol'] = 1e-3

        ikey = 0
        for key in desvars:
            model.add_design_var(key, lower=bounds[ikey,0], upper=bounds[ikey,1])
            ikey+=1

        model.add_constraint('GenSpeed_Max', upper=1.2)
        model.add_objective('TwrBsMyt_DEL', ref=1.e0)
        p.driver.recording_options['includes'] = ['*']
        p.driver.recording_options['record_objectives'] = True
        p.driver.recording_options['record_constraints'] = True
        p.driver.recording_options['record_desvars'] = True
        p.driver.recording_options['record_inputs'] = True
        p.driver.recording_options['record_outputs'] = True
        p.driver.recording_options['record_residuals'] = True

        p.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs','totals']

        recorder = om.SqliteRecorder("cases.sql")
        p.driver.add_recorder(recorder)

        p.setup()
        p.run_driver()

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
