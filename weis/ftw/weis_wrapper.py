import numpy as np
import os, sys
import warnings
import openmdao.api as om
from copy                               import copy
from weis.glue_code.gc_LoadInputs       import WindTurbineOntologyPythonWEIS
from weis.glue_code.gc_PoseOptimization import PoseOptimizationWEIS
from openmdao.utils.mpi                 import MPI
from wisdem.inputs                      import load_yaml
from openfast_io.FileTools              import save_yaml
from wisdem.inputs.validation           import simple_types
from weis                               import weis_main


def ftw_doe(fname_wt_input, fname_modeling_options, fname_opt_options, geometry_override=None, modeling_override=None, analysis_override=None, test_run=False):

    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options,
        modeling_override=modeling_override,
        analysis_override=analysis_override
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    # Files and paths
    if os.path.isabs(opt_options['general']['folder_output']):
        out_dir = opt_options['general']['folder_output']
    else:
        out_dir = os.path.join(os.path.realpath(os.curdir), opt_options['general']['folder_output'])
    if os.path.isabs(opt_options['recorder']['file_name']):
        recorder_output_path = opt_options['recorder']['file_name']
    else:
        recorder_output_path = os.path.join(out_dir, opt_options['recorder']['file_name'])
    fname_doedata = recorder_output_path + '-doedata.yaml'
    fname_sm = recorder_output_path + '-surrogate.smt'

    if MPI:
        recorder_output_path += ('_' + str(int(rank)))

    # If skip flag, check existence of the doedata file, read, and broadcast to all MPI cores
    doedata = None
    if (not MPI) or (MPI and rank == 0):
        if opt_options['driver']['design_of_experiments']['skip_doe_if_doedata_exist']:
            if os.path.isfile(os.path.join(out_dir, fname_doedata)):
                try:
                    doedata = load_yaml(os.path.join(out_dir, fname_doedata))
                    print('{:} file exists. Skipping design of experiments.'.format(fname_doedata))
                except:
                    doedata = None
                    print('Unable to read {:} file. Executing design of experiments.'.format(fname_doedata))
            else:
                print('{:} file does not exist. Executing design of experiments.'.format(fname_doedata))
    if opt_options['driver']['design_of_experiments']['skip_training_if_sm_exist']:
        skip_training_if_sm_exist = True
    else:
        skip_training_if_sm_exist = False
    if MPI:
        doedata = MPI.COMM_WORLD.bcast(doedata, root=0)
    if doedata != None:
        return doedata, fname_doedata, fname_sm, skip_training_if_sm_exist

    # Raise error when a proper simulation level is not specified.
    if modeling_options['OpenFAST']['flag']:
        raise NotImplementedError('Design coupling analysis with OpenFAST response is not implemented yet.')
    elif modeling_options['OpenFAST_Linear']['flag']:
        raise NotImplementedError('Design coupling analysis with OpenFAST_Linear response is not implemented yet.')
    elif modeling_options['RAFT']['flag']:
        pass
    else:
        raise Exception('A simulation level (RAFT, OpenFAST_Linear, OpenFAST) should be enabled for design coupling analysis.')

    # Raise error when the recorder is disabled or just_dvs is enabled
    if not opt_options['recorder']['flag']:
        raise Exception('Recorder option should be enabled to save DOE results.')
    if opt_options['recorder']['just_dvs']:
        raise Exception('Recorder should include output variables in addition to design variables.')

    # Merge recorder variables
    opt_override = {}
    opt_override['recorder'] = {}
    opt_override['recorder']['includes'] = opt_options['recorder']['includes'] + opt_options['recorder']['surrogate_model_outputs']

    # Enable design of experiment driver to obtain training sample data
    opt_override['driver'] = {}
    opt_override['driver']['optimization'] = {}
    opt_override['driver']['optimization']['flag'] = False
    opt_override['driver']['design_of_experiments'] = {}
    opt_override['driver']['design_of_experiments']['flag'] = True
    if opt_options['driver']['design_of_experiments']['generator'] != 'LatinHypercube':
        warnings.warn('LatinHypercube generator with sufficient number of samples is recommended.')

    # Run DOE (in parallel if MPI is used)
    wt_opt_doe, modeling_options_doe, opt_options_doe = weis_main(
        fname_wt_input, fname_modeling_options, fname_opt_options,
        geometry_override, {}, opt_override, test_run)

    # OpenMDAO case reader - obtain list of cases
    cr = om.CaseReader(recorder_output_path)
    cases = cr.list_cases(source = 'driver', out_stream = None)

    # Obtain list of key variables (only at rank 0, will be broadcasted later)
    if (not MPI) or (MPI and rank == 0):
        # Choose the first case from rank 0
        case = cr.get_case(cases[0])

        # Design variables (inputs of the surrogate model)
        design_vars = list(case.get_design_vars().keys())
        # Constraint variables (outputs of the surrogate model)
        constraint_vars = list(case.get_constraints().keys())
        # Objective variables (outputs of the surrogate model)
        objective_vars = list(case.get_objectives().keys())
        # Additional output variables (outputs of the surrogate model)
        additional_output_vars = opt_options['recorder']['surrogate_model_outputs']

        # Combine list of variables (combine and remain unique items)
        all_vars = list(set(design_vars + constraint_vars + objective_vars + additional_output_vars))
        # Split variables into inputs and outputs of surrogate model
        input_vars = design_vars
        input_vars.sort()
        output_vars = list(set(all_vars) - set(design_vars))
        output_vars.sort()

        # Determine length of each variable and set how to treat vectors
        input_lens = []
        input_vecs = []
        output_lens = []
        output_vecs = []
        for idx in range(len(input_vars)):
            l = case.get_val(input_vars[idx]).size
            input_lens.append(l)
            if l == 1:
                input_vecs.append(None)
            else:
                input_vecs.append('vector')

        for idx in range(len(output_vars)):
            l = case.get_val(output_vars[idx]).size
            output_lens.append(l)
            if l == 1:
                output_vecs.append(None) # Scalar variable
            elif output_vars[idx] in constraint_vars: # If constraint
                varname = output_vars[idx]
                lb = wt_opt_doe.model.get_constraints()[varname]['lower']
                if lb <= -1.0e20:
                    lb = None # No lower bound
                ub = wt_opt_doe.model.get_constraints()[varname]['upper']
                if ub >= +1.0e20:
                    ub = None # No upper bound
                # min if lb, max if ub, vector if both, error if none
                if lb and (not ub):
                    output_vecs.append('min')
                    output_lens[idx] = 1
                elif (not lb) and ub:
                    output_vecs.append('max')
                    output_lens[idx] = 1
                elif lb and ub:
                    output_vecs.append('vector')
                else:
                    raise Exception('Constraint unbounded.')
            else:
                output_vecs.append('vector')
    else:
        # rank != 0
        input_vars = []
        input_lens = []
        input_vecs = []
        output_vars = []
        output_lens = []
        output_vecs = []

    # Broadcast list of variables, their lengths, and vector treatments
    # to all MPI instances
    if MPI:
        input_vars = MPI.COMM_WORLD.bcast(input_vars, root=0)
        input_lens = MPI.COMM_WORLD.bcast(input_lens, root=0)
        input_vecs = MPI.COMM_WORLD.bcast(input_vecs, root=0)
        output_vars = MPI.COMM_WORLD.bcast(output_vars, root=0)
        output_lens = MPI.COMM_WORLD.bcast(output_lens, root=0)
        output_vecs = MPI.COMM_WORLD.bcast(output_vecs, root=0)

    # Empty dataset list
    input_dataset = [np.zeros((0, input_lens[idx]), dtype=float) for idx in range(len(input_vars))]
    output_dataset = [np.zeros((0, output_lens[idx]), dtype=float) for idx in range(len(output_vars))]
    
    # Retrieve data from each case for all cases and all MPI instances
    for casename in cases:
        case = cr.get_case(casename)
        for idx in range(len(input_vars)):
            data_currentcase = np.array([case.get_val(input_vars[idx]).flatten()])
            input_dataset[idx] = np.concatenate(
                (input_dataset[idx], data_currentcase), axis=0)
        for idx in range(len(output_vars)):
            data_currentcase = np.array([case.get_val(output_vars[idx]).flatten()])
            if data_currentcase.size == output_lens[idx]:
                output_dataset[idx] = np.concatenate(
                    (output_dataset[idx], data_currentcase), axis=0)
            else:
                if output_vecs[idx].lower() == 'min':
                    output_dataset[idx] = np.concatenate(
                        (output_dataset[idx], np.min(data_currentcase).reshape(1,1)), axis=0)
                elif output_vecs[idx].lower() == 'max':
                    output_dataset[idx] = np.concatenate(
                        (output_dataset[idx], np.max(data_currentcase).reshape(1,1)), axis=0)
                else:
                    raise Exception('Unknown vector treatment or dimension mismatch.')

    # If MPI, gather data to rank=0
    if MPI:
        input_dataset_gathered = MPI.COMM_WORLD.gather(input_dataset, root=0)
        output_dataset_gathered = MPI.COMM_WORLD.gather(output_dataset, root=0)
    else:
        input_dataset_gathered = [copy(input_dataset)]
        output_dataset_gathered = [copy(output_dataset)]
    
    if (not MPI) or (MPI and rank == 0):
        # Empty dataset list
        input_dataset = [np.zeros((0, input_lens[idx]), dtype=float) for idx in range(len(input_vars))]
        output_dataset = [np.zeros((0, output_lens[idx]), dtype=float) for idx in range(len(output_vars))]
        
        # Merge input dataset from all MPI instances
        for idx in range(len(input_dataset_gathered)):
            for jdx in range(len(input_dataset_gathered[0])):
                data_currentcase = input_dataset_gathered[idx][jdx]
                input_dataset[jdx] = np.concatenate(
                    (input_dataset[jdx], data_currentcase), axis=0)

        # Merge output dataset from all MPI instances
        for idx in range(len(output_dataset_gathered)):
            for jdx in range(len(output_dataset_gathered[0])):
                data_currentcase = output_dataset_gathered[idx][jdx]
                output_dataset[jdx] = np.concatenate(
                    (output_dataset[jdx], data_currentcase), axis=0)

        # Convert numpy arrays to nested list
        for idx in range(len(input_dataset)):
            if type(input_dataset[idx]) == np.ndarray:
                input_dataset[idx] = input_dataset[idx].tolist()
        for idx in range(len(output_dataset)):
            if type(output_dataset[idx]) == np.ndarray:
                output_dataset[idx] = output_dataset[idx].tolist()
    else:
        input_dataset = None
        output_dataset = None

    # Save data into a single yaml file
    if (not MPI) or (MPI and rank == 0):
        doedata = {
            'input': [],
            'output': [],
        }

        # Input variables
        for idx in range(len(input_vars)):
            doedata['input'].append({
                'name': input_vars[idx],
                'data': input_dataset[idx],
                'len': input_lens[idx],
                'vec': input_vecs[idx]})

        # Output variables
        for idx in range(len(output_vars)):
            doedata['output'].append({
                'name': output_vars[idx],
                'data': output_dataset[idx],
                'len': output_lens[idx],
                'vec': output_vecs[idx]})

        # Save to yaml file: [output-folder]/[output-name]-doedata.yaml
        save_yaml(out_dir, fname_doedata, doedata)

    return doedata, fname_doedata, fname_sm, skip_training_if_sm_exist

