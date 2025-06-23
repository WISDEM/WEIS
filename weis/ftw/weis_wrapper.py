import numpy as np
import os, sys
import warnings
import openmdao.api as om
from weis.glue_code.gc_LoadInputs       import WindTurbineOntologyPythonWEIS
from weis.glue_code.gc_PoseOptimization import PoseOptimizationWEIS
from openmdao.utils.mpi                 import MPI
from openfast_io.FileTools              import save_yaml
from wisdem.inputs.validation           import simple_types
from weis import weis_main


def ftw_doe(fname_wt_input, fname_modeling_options, fname_opt_options, geometry_override=None, modeling_override=None, analysis_override=None, test_run=False):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options,
        modeling_override=modeling_override,
        analysis_override=analysis_override
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

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

    # Enable design of experiment driver to obtain training sample data
    opt_options['driver']['optimization']['flag'] = False
    opt_options['driver']['design_of_experiments']['flag'] = True
    if opt_options['driver']['design_of_experiments']['generator'] != 'LatinHypercube':
        warnings.warn('LatinHypercube generator with sufficient number of samples is recommended.')

    # Run DOE (in parallel if MPI is used)
    wt_opt_doe, modeling_options_doe, opt_options_doe = weis_main(
        fname_wt_input, fname_modeling_options, fname_opt_options,
        geometry_override, modeling_override, analysis_override, test_run)

    # Read SQL file(s)

    return wt_opt_doe, modeling_options_doe, opt_options_doe


def ftw_training(fname_wt_input, fname_modeling_options, fname_opt_options, run_dir, geometry_override=None, modeling_override=None, analysis_override=None):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options,
        modeling_override=modeling_override,
        analysis_override=analysis_override
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    # Output folder and recorder output file paths
    if os.path.isabs(opt_options['general']['folder_output']):
        out_dir = opt_options['general']['folder_output']
    else:
        out_dir = os.path.join(run_dir, opt_options['general']['folder_output'])
    if os.path.isabs(opt_options['recorder']['file_name']):
        recorder_output_path = opt_options['recorder']['file_name']
    else:
        recorder_output_path = os.path.join(out_dir, opt_options['recorder']['file_name'])

    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        recorder_output_path += '_' + str(int(rank)) # MPI SQL files have rank numbers
    else:
        rank = 0

    cr = om.CaseReader(recorder_output_path)
    cases = cr.list_cases(source = 'driver', out_stream = None)

    # Obtain list of key variables
    if (not MPI) or (MPI and rank == 0):
        case = cr.get_case(cases[0])
        # Design variables (inputs of the surrogate model)
        design_vars = list(case.get_design_vars().keys())
        n_design_vars = len(design_vars)
        # Design variables (outputs of the surrogate model)
        constraint_vars = list(case.get_constraints().keys())
        n_constraint_vars = len(constraint_vars)













