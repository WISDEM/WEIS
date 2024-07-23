import numpy as np
import os, sys, time, json
import openmdao.api as om
from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel import yaml2openmdao
from weis.glue_code.gc_PoseOptimization  import PoseOptimizationWEIS
from weis.glue_code.glue_code         import WindPark
from wisdem.commonse.mpi_tools        import MPI
from wisdem.commonse                  import fileIO
from weis.glue_code.gc_ROSCOInputs    import assign_ROSCO_values
from weis.control.tmd                 import assign_TMD_values
from weis.surrogate.WTsurrogate       import WindTurbineDOE2SM


fd_methods = ['SLSQP','SNOPT', 'LD_MMA']
crawling_methods = ['DE', 'NSGA2']

if MPI:
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop

def run_weis(fname_wt_input, fname_modeling_options, fname_opt_options, geometry_override=None, modeling_override=None, analysis_override=None):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options,
        modeling_override=modeling_override,
        analysis_override=analysis_override
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    # Initialize openmdao problem. If running with multiple processors in MPI, use parallel finite differencing equal to the number of cores used.
    # Otherwise, initialize the WindPark system normally. Get the rank number for parallelization. We only print output files using the root processor.
    myopt = PoseOptimizationWEIS(wt_init, modeling_options, opt_options)

    if MPI:
        n_DV = myopt.get_number_design_variables()
        # Extract the number of cores available
        max_cores = MPI.COMM_WORLD.Get_size()

        # Define the color map for the parallelization, determining the maximum number of parallel finite difference (FD)
        # evaluations based on the number of design variables (DV). OpenFAST on/off changes things.
        if modeling_options['Level3']['flag']:

            # If we are running an optimization method that doesn't use finite differencing, set the number of DVs to 1
            if not (opt_options['driver']['design_of_experiments']['flag'] or opt_options['driver']['optimization']['solver'] in fd_methods):
                n_DV = 1

            # If openfast is called, the maximum number of FD is the number of DV, if we have the number of cores available that doubles the number of DVs,
            # otherwise it is half of the number of DV (rounded to the lower integer).
            # We need this because a top layer of cores calls a bottom set of cores where OpenFAST runs.
            if max_cores > 2. * n_DV:
                n_FD = n_DV
            else:
                n_FD = int(np.floor(max_cores / 2))
            # Get the number of OpenFAST runs from the user input and the max that can run in parallel given the resources
            # The number of OpenFAST runs is the minimum between the actual number of requested OpenFAST simulations, and
            # the number of cores available (minus the number of DV, which sit and wait for OF to complete)
            n_OF_runs = modeling_options['DLC_driver']['n_cases']
            n_DV = max([n_DV, 1])
            max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
            n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])
        elif modeling_options['Level2']['flag']:

            if not (opt_options['driver']['design_of_experiments']['flag'] or opt_options['driver']['optimization']['solver'] in fd_methods):
                n_DV = 1


            if max_cores > 2. * n_DV:
                n_FD = n_DV
            else:
                n_FD = int(np.floor(max_cores / 2))
            n_OF_runs = modeling_options['Level2']['linearization']['NLinTimes']
            n_DV = max([n_DV, 1])
            max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
            n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])
        else:
            # If OpenFAST is not called, the number of parallel calls to compute the FDs is just equal to the minimum of cores available and DV
            n_FD = min([max_cores, n_DV])
            n_OF_runs_parallel = 1
            # if we're doing a GA or such, "FD" means "entities in epoch"
            if opt_options['driver']['optimization']['solver'] in crawling_methods:
                n_FD = max_cores

        # Define the color map for the cores (how these are distributed between finite differencing and openfast runs)
        if opt_options['driver']['design_of_experiments']['flag']:
            n_FD = MPI.COMM_WORLD.Get_size()
            n_OF_runs_parallel = 1
            rank    = MPI.COMM_WORLD.Get_rank()
            comm_map_up = comm_map_down = {}
            for r in range(MPI.COMM_WORLD.Get_size()):
                comm_map_up[r] = [r]
            color_i = 0
        else:
            n_FD = max([n_FD, 1])
            if modeling_options['Level3']['flag'] == True and modeling_options['Level3']['AeroDyn']['WakeMod'] == 3:
                olaf = True
            else:
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

    # make the folder_output relative to the input, if it's a relative path
    analysis_input_dir = os.path.dirname(opt_options['fname_input_analysis'])
    opt_options['general']['folder_output'] = os.path.join(analysis_input_dir,opt_options['general']['folder_output'])

    folder_output = opt_options['general']['folder_output']
    if rank == 0 and not os.path.isdir(folder_output):
        os.makedirs(folder_output,exist_ok=True)

    if color_i == 0: # the top layer of cores enters, the others sit and wait to run openfast simulations
        # if MPI and opt_options['driver']['optimization']['flag']:
        if MPI:
            if modeling_options['Level3']['flag'] or modeling_options['Level2']['flag']:
                # Parallel settings for OpenFAST
                modeling_options['General']['openfast_configuration']['mpi_run'] = True
                modeling_options['General']['openfast_configuration']['mpi_comm_map_down'] = comm_map_down
                if opt_options['driver']['design_of_experiments']['flag']:
                    modeling_options['General']['openfast_configuration']['cores'] = 1
                else:
                    modeling_options['General']['openfast_configuration']['cores'] = n_OF_runs_parallel

            # Parallel settings for OpenMDAO
            if opt_options['driver']['design_of_experiments']['flag']:
                wt_opt = om.Problem(model=WindPark(modeling_options = modeling_options, opt_options = opt_options), reports=False)
            else:
                wt_opt = om.Problem(model=om.Group(num_par_fd=n_FD), comm=comm_i, reports=False)
                wt_opt.model.add_subsystem('comp', WindPark(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        else:
            # Sequential finite differencing and openfast simulations
            modeling_options['General']['openfast_configuration']['mpi_run'] = False
            modeling_options['General']['openfast_configuration']['cores']   = 1
            wt_opt = om.Problem(model=WindPark(modeling_options = modeling_options, opt_options = opt_options), reports=False)

        # If at least one of the design variables is active, setup an optimization
        if opt_options['opt_flag']:
            wt_opt = myopt.set_driver(wt_opt)
            wt_opt = myopt.set_objective(wt_opt)
            wt_opt = myopt.set_design_variables(wt_opt, wt_init)
            wt_opt = myopt.set_constraints(wt_opt)

            if opt_options['driver']['design_of_experiments']['flag']:
                wt_opt.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs']
                wt_opt.driver.options['procs_per_model'] = 1 # n_OF_runs_parallel # int(max_cores / np.floor(max_cores/n_OF_runs))

        wt_opt = myopt.set_recorders(wt_opt)
        wt_opt.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs','totals']

        # Setup openmdao problem
        if opt_options['opt_flag']:
            wt_opt.setup()
        else:
            # If we're not performing optimization, we don't need to allocate
            # memory for the derivative arrays.
            wt_opt.setup(derivatives=False)

        # Load initial wind turbine data from wt_initial to the openmdao problem
        wt_opt = yaml2openmdao(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt = assign_ROSCO_values(wt_opt, modeling_options, opt_options)
        if modeling_options['flags']['TMDs']:
            wt_opt = assign_TMD_values(wt_opt, wt_init, opt_options)

        wt_opt = myopt.set_initial(wt_opt, wt_init)
        if modeling_options['Level3']['flag']:
            wt_opt = myopt.set_initial_weis(wt_opt)

        # If the user provides values in geometry_override, they overwrite
        # whatever values have been set by the yaml files.
        # This is useful for performing black-box wrapped optimization without
        # needing to modify the yaml files.
        # Some logic is used here if the user gives a smalller size for the
        # design variable than expected to input the values into the end
        # of the array.
        # This is useful when optimizing twist, where the first few indices
        # do not need to be optimized as they correspond to a circular cross-section.
        if geometry_override is not None:
            for key in geometry_override:
                num_values = np.array(geometry_override[key]).size
                key_size = wt_opt[key].size
                idx_start = key_size - num_values
                wt_opt[key][idx_start:] = geometry_override[key]

        # Place the last design variables from a previous run into the problem.
        # This needs to occur after the above setup() and yaml2openmdao() calls
        # so these values are correctly placed in the problem.
        wt_opt = myopt.set_restart(wt_opt)

        if 'check_totals' in opt_options['driver']['optimization']:
            if opt_options['driver']['optimization']['check_totals']:
                wt_opt.run_model()
                totals = wt_opt.compute_totals()

        if 'check_partials' in opt_options['driver']['optimization']:
            if opt_options['driver']['optimization']['check_partials']:
                wt_opt.run_model()
                checks = wt_opt.check_partials(compact_print=True)

        sys.stdout.flush()

        # Check if DOE & skip flag True & DOE results exist, skip DOE
        SKIP_DRIVER = False
        sm_filename = os.path.join(folder_output, os.path.splitext(opt_options['recorder']['file_name'])[0] + '.smt')
        if opt_options['opt_flag']:
            if opt_options['driver']['design_of_experiments']['flag']:
                if 'skip_if_results_exist' in opt_options['driver']['design_of_experiments']:
                    if opt_options['driver']['design_of_experiments']['skip_if_results_exist']:
                        if os.path.isfile(sm_filename):
                            SKIP_DRIVER = True
                            if (not MPI) or (MPI and rank == 0):
                                print('File {:} exists. Skipping the design of experiments.'.format(sm_filename))

        # Run openmdao problem
        if opt_options['opt_flag']:
            if not SKIP_DRIVER:
                wt_opt.run_driver()
        else:
            wt_opt.run_model()

        if ((not MPI) or (MPI and rank == 0)) and (not SKIP_DRIVER):
            # Save data coming from openmdao to an output yaml file
            froot_out = os.path.join(folder_output, opt_options['general']['fname_output'])
            # Remove the fst_vt key from the dictionary and write out the modeling options
            modeling_options['General']['openfast_configuration']['fst_vt'] = {}
            if not modeling_options['Level3']['from_openfast']:
                wt_initial.write_ontology(wt_opt, froot_out)
            wt_initial.write_options(froot_out)

            # output the problem variables as a dictionary in the output dir
            fname_pv_json = os.path.join(folder_output, "problem_vars.json")
            pvfile = open(fname_pv_json, 'w')
            # openMDAO doesn't save constraint values, so we get them from this construction
            problem_var_dict = wt_opt.list_driver_vars(
                desvar_opts=["lower", "upper",],
                cons_opts=["lower", "upper", "equals",],
                out_stream=pvfile,
            )
            pvfile.close()
            
            # clean up the problem_var_dict that we extracted for output
            for k in problem_var_dict.keys():
                if not problem_var_dict.get(k): continue
                for idx in range(len(problem_var_dict[k])):
                    for kk in problem_var_dict[k][idx][1].keys():
                        if isinstance(problem_var_dict[k][idx][1][kk], np.ndarray):
                            problem_var_dict[k][idx][1][kk] = problem_var_dict[k][idx][1][kk].tolist()
                        if isinstance(problem_var_dict[k][idx][1][kk], np.int32):
                            problem_var_dict[k][idx][1][kk] = int(problem_var_dict[k][idx][1][kk])
            #with open(fname_pv_json, 'w') as pvfile:
            #    json.dump(problem_var_dict, pvfile, indent=4)

            # Save data to numpy and matlab arrays
            fileIO.save_data(froot_out, wt_opt)

    if MPI and \
            (modeling_options['Level3']['flag'] or modeling_options['Level2']['flag']) and \
            (not opt_options['driver']['design_of_experiments']['flag']) and \
            color_i < 1000000:
        # subprocessor ranks spin, waiting for FAST simulations to run.
        # Only true for cores actually in use, not the ones supporting openfast openmp (marked as color_i = 1000000)
        sys.stdout.flush()
        if rank in comm_map_up.keys():
            subprocessor_loop(comm_map_up)
        sys.stdout.flush()

        # close signal to subprocessors
        subprocessor_stop(comm_map_down)
        sys.stdout.flush()

    # Send each core in use to a barrier synchronization
    if MPI and color_i < 1000000:
        MPI.COMM_WORLD.Barrier()

    # If design_of_experiment, recorder flag, train_surrogate_model are all True,
    # collect sql files and create smt object
    if opt_options['opt_flag'] and (not SKIP_DRIVER):
        if opt_options['driver']['design_of_experiments']['flag'] and opt_options['recorder']['flag']:
            if opt_options['driver']['design_of_experiments']['train_surrogate_model']:
                sql_file = os.path.join(folder_output, opt_options['recorder']['file_name'])
                if MPI:
                    sql_file += '_{:}'.format(rank)
                WTSM = WindTurbineDOE2SM()
                WTSM.read_doe(sql_file, modeling_options, opt_options) # Parallel reading if MPI
                WTSM.train_sm() # Parallel training if MPI
                WTSM.write_sm(sm_filename) # Saving will be done in rank=0

    if rank == 0:
        return wt_opt, modeling_options, opt_options
    else:
        return [], [], []
