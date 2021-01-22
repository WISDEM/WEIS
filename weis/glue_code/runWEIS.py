import numpy as np
import os, sys, time
import openmdao.api as om
from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel import yaml2openmdao
from weis.glue_code.gc_PoseOptimization  import PoseOptimizationWEIS
from weis.glue_code.glue_code         import WindPark
from wisdem.commonse.mpi_tools        import MPI
from wisdem.commonse                  import fileIO
from weis.glue_code.gc_ROSCOInputs    import assign_ROSCO_values

if MPI:
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop

def run_weis(fname_wt_input, fname_modeling_options, fname_opt_options, overridden_values=None):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_opt_options)
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    # Initialize openmdao problem. If running with multiple processors in MPI, use parallel finite differencing equal to the number of cores used.
    # Otherwise, initialize the WindPark system normally. Get the rank number for parallelization. We only print output files using the root processor.
    myopt = PoseOptimizationWEIS(modeling_options, opt_options)

    if MPI:
        n_DV = myopt.get_number_design_variables()
        
        # Extract the number of cores available
        max_cores = MPI.COMM_WORLD.Get_size()

        # Define the color map for the parallelization, determining the maximum number of parallel finite difference (FD) evaluations based on the number of design variables (DV). OpenFAST on/off changes things.
        if modeling_options['Level3']['flag']:
            # If openfast is called, the maximum number of FD is the number of DV, if we have the number of cores available that doubles the number of DVs, otherwise it is half of the number of DV (rounded to the lower integer). We need this because a top layer of cores calls a bottom set of cores where OpenFAST runs.
            if max_cores > 2. * n_DV:
                n_FD = n_DV
            else:
                n_FD = int(np.floor(max_cores / 2))
            # The number of OpenFAST runs is the minimum between the actual number of requested OpenFAST simulations, and the number of cores available (minus the number of DV, which sit and wait for OF to complete)
            
            # need to calculate the number of OpenFAST runs from the user input
            n_OF_runs = 0
            if modeling_options['openfast']['dlc_settings']['run_power_curve']:
                if modeling_options['openfast']['dlc_settings']['Power_Curve']['turbulent_power_curve']:
                    n_OF_runs += len(modeling_options['openfast']['dlc_settings']['Power_Curve']['U'])*len(modeling_options['openfast']['dlc_settings']['Power_Curve']['Seeds'])
                else:
                    n_OF_runs += len(modeling_options['openfast']['dlc_settings']['Power_Curve']['U'])
            if modeling_options['openfast']['dlc_settings']['run_IEC'] or modeling_options['openfast']['dlc_settings']['run_blade_fatigue']:
                for dlc in modeling_options['openfast']['dlc_settings']['IEC']:
                    dlc_vars = list(dlc.keys())
                    # Number of wind speeds
                    if 'U' not in dlc_vars:
                        if dlc['DLC'] == 1.4: # assuming 1.4 is run at [V_rated-2, V_rated, V_rated] and +/- direction change
                            n_U = 6
                        elif dlc['DLC'] == 5.1: # assuming 1.4 is run at [V_rated-2, V_rated, V_rated]
                            n_U = 3
                        elif dlc['DLC'] in [6.1, 6.3]: # assuming V_50 for [-8, 8] deg yaw error
                            n_U = 2
                        else:
                            print('Warning: for OpenFAST DLC %1.1f specified in the Analysis Options, wind speeds "U" must be provided'%dlc['DLC'])
                    else:
                        n_U = len(dlc['U'])
                    # Number of seeds
                    if 'Seeds' not in dlc_vars:
                        if dlc['DLC'] == 1.4: # not turbulent
                            n_Seeds = 1
                        else:
                            print('Warning: for OpenFAST DLC %1.1f specified in the Analysis Options, turbulent seeds "Seeds" must be provided'%dlc['DLC'])
                    else:
                        n_Seeds = len(dlc['Seeds'])

                    n_OF_runs += n_U*n_Seeds

            n_DV = max([n_DV, 1])
            max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
            n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])

            modeling_options['openfast']['dlc_settings']['n_OF_runs'] = n_OF_runs
        else:
            # If OpenFAST is not called, the number of parallel calls to compute the FDs is just equal to the minimum of cores available and DV
            n_FD = min([max_cores, n_DV])
            n_OF_runs_parallel = 1
        
        # Define the color map for the cores (how these are distributed between finite differencing and openfast runs)
        n_FD = max([n_FD, 1])
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(n_FD, n_OF_runs_parallel)
        rank    = MPI.COMM_WORLD.Get_rank()
        color_i = color_map[rank]
        comm_i  = MPI.COMM_WORLD.Split(color_i, 1)
    else:
        color_i = 0
        rank = 0

    folder_output = opt_options['general']['folder_output']
    if rank == 0 and not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    if color_i == 0: # the top layer of cores enters, the others sit and wait to run openfast simulations
        if MPI:
            if modeling_options['Level3']['flag']:
                # Parallel settings for OpenFAST
                modeling_options['openfast']['analysis_settings']['mpi_run']           = True
                modeling_options['openfast']['analysis_settings']['mpi_comm_map_down'] = comm_map_down
                modeling_options['openfast']['analysis_settings']['cores']             = n_OF_runs_parallel            
            # Parallel settings for OpenMDAO
            wt_opt = om.Problem(model=om.Group(num_par_fd=n_FD), comm=comm_i)
            wt_opt.model.add_subsystem('comp', WindPark(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        else:
            # Sequential finite differencing and openfast simulations
            modeling_options['openfast']['analysis_settings']['cores'] = 1
            wt_opt = om.Problem(model=WindPark(modeling_options = modeling_options, opt_options = opt_options))

        # If at least one of the design variables is active, setup an optimization
        if opt_options['opt_flag']:
            wt_opt = myopt.set_driver(wt_opt)
            wt_opt = myopt.set_objective(wt_opt)
            wt_opt = myopt.set_design_variables(wt_opt, wt_init)
            wt_opt = myopt.set_constraints(wt_opt)
            wt_opt = myopt.set_recorders(wt_opt)
        
        # Setup openmdao problem
        if opt_options['opt_flag']:
            wt_opt.setup()
        else:
            # If we're not performing optimization, we don't need to allocate
            # memory for the derivative arrays.
            wt_opt.setup(derivatives=False)
        
        # Load initial wind turbine data from wt_initial to the openmdao problem
        wt_opt = yaml2openmdao(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt = assign_ROSCO_values(wt_opt, modeling_options, wt_init['control'])
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        
        # If the user provides values in this dict, they overwrite
        # whatever values have been set by the yaml files.
        # This is useful for performing black-box wrapped optimization without
        # needing to modify the yaml files.
        # Some logic is used here if the user gives a smalller size for the
        # design variable than expected to input the values into the end
        # of the array.
        # This is useful when optimizing twist, where the first few indices
        # do not need to be optimized as they correspond to a circular cross-section.
        if overridden_values is not None:
            for key in overridden_values:
                num_values = np.array(overridden_values[key]).size
                key_size = wt_opt[key].size
                idx_start = key_size - num_values
                wt_opt[key][idx_start:] = overridden_values[key]

        # Place the last design variables from a previous run into the problem.
        # This needs to occur after the above setup() and yaml2openmdao() calls
        # so these values are correctly placed in the problem.
        wt_opt = myopt.set_restart(wt_opt)

        if 'check_totals' in opt_options['driver']:
            if opt_options['driver']['check_totals']:
                wt_opt.run_model()
                totals = wt_opt.compute_totals()
        
        if 'check_partials' in opt_options['driver']:
            if opt_options['driver']['check_partials']:
                wt_opt.run_model()
                checks = wt_opt.check_partials(compact_print=True)
                
        sys.stdout.flush()
        # Run openmdao problem
        if opt_options['opt_flag']:
            wt_opt.run_driver()
        else:
            wt_opt.run_model()

        if (not MPI) or (MPI and rank == 0):
            # Save data coming from openmdao to an output yaml file
            froot_out = os.path.join(folder_output, opt_options['general']['fname_output'])
            wt_initial.update_ontology_control(wt_opt)
            # Remove the fst_vt key from the dictionary and write out the modeling options
            modeling_options['openfast']['fst_vt'] = {}
            wt_initial.write_ontology(wt_opt, froot_out)
            wt_initial.write_options(froot_out)
            
            # Save data to numpy and matlab arrays
            fileIO.save_data(froot_out, wt_opt)

    if MPI and modeling_options['Level3']['flag']:
        # subprocessor ranks spin, waiting for FAST simulations to run
        sys.stdout.flush()
        if rank in comm_map_up.keys():
            subprocessor_loop(comm_map_up)
        sys.stdout.flush()

        # close signal to subprocessors
        subprocessor_stop(comm_map_down)
        sys.stdout.flush()

        
    if rank == 0:
        return wt_opt, modeling_options, opt_options
    else:
        return [], [], []
